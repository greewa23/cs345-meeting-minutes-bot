import os
import re
import json
import tempfile
import subprocess
import threading
from io import BytesIO
from datetime import datetime, date, timedelta
import uuid

from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
import requests


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)

# Maximum upload size: 25 MB
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")

# Supported candidate languages for automatic detection
SUPPORTED_LANGUAGES = [
    "en-US",
    "es-ES",
    "fr-FR",
    "de-DE"
]

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".webm",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac"
}

LANGUAGE_API_VERSION = "2023-04-01"


# ============================================================
# 2. AUDIO CONVERSION
# ============================================================

def check_ffmpeg():
    """Verify FFmpeg is installed and accessible."""

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10
        )

    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "FFmpeg is not installed or configured correctly."
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "FFmpeg availability check timed out."
        )


def convert_to_wav(input_path, output_path):
    """
    Convert uploaded audio to 16 kHz, 16-bit,
    mono PCM WAV for Azure Speech.
    """

    check_ffmpeg()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-map", "0:a:0",
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                output_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=120
        )

    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed.")

    except subprocess.CalledProcessError as error:
        raise ValueError(
            "Unsupported or invalid audio format: "
            + error.stderr[-1000:]
        )

    except subprocess.TimeoutExpired:
        raise ValueError("Audio conversion timed out.")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ValueError("Audio conversion produced an empty WAV file.")


def ticks_to_seconds(value):
    """Convert Azure 100-nanosecond ticks into seconds."""

    try:
        return int(value) / 10_000_000
    except (TypeError, ValueError):
        return 0.0

# ============================================================
# AZURE SPEECH-TO-TEXT TRANSCRIPTION
# ============================================================



def transcribe_audio(audio_path):

    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        raise RuntimeError("Azure Speech credentials are missing.")

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )

    speech_config.output_format = speechsdk.OutputFormat.Detailed

    auto_detect_config = (
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=SUPPORTED_LANGUAGES
        )
    )

    audio_config = speechsdk.audio.AudioConfig(
        filename=audio_path
    )

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_config,
        audio_config=audio_config
    )

    results = []
    error_messages = []
    detected_language = None

    finished_event = threading.Event()

    def recognized_callback(evt):

        nonlocal detected_language

        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return

        if not evt.result.text.strip():
            return

        try:
            language_result = (
                speechsdk.AutoDetectSourceLanguageResult(evt.result)
            )

            if language_result.language:
                detected_language = language_result.language

        except Exception:
            pass

        words = []

        # Extract detailed word-level recognition results.
        try:
            detailed_result = json.loads(evt.result.json)

            nbest = detailed_result.get("NBest", [])

            if nbest:
                for word in nbest[0].get("Words", []):

                    words.append({
                        "word": word.get("Word"),
                        "offset": round(
                            ticks_to_seconds(word.get("Offset", 0)),
                            3
                        ),
                        "duration": round(
                            ticks_to_seconds(word.get("Duration", 0)),
                            3
                        ),
                        "confidence": word.get("Confidence")
                    })

        except (ValueError, TypeError, KeyError):
            pass

        results.append({
            "text": evt.result.text,
            "offset": ticks_to_seconds(evt.result.offset),
            "duration": ticks_to_seconds(evt.result.duration),
            "words": words
        })

    def canceled_callback(evt):

        error_messages.append(
            evt.result.error_details or
            "Speech recognition was canceled."
        )

        finished_event.set()

    def session_stopped_callback(evt):
        finished_event.set()

    recognizer.recognized.connect(recognized_callback)
    recognizer.canceled.connect(canceled_callback)
    recognizer.session_stopped.connect(session_stopped_callback)

    try:

        # Start continuous recognition asynchronously.
        recognizer.start_continuous_recognition_async().get()

        if not finished_event.wait(timeout=3600):
            raise TimeoutError(
                "Speech recognition exceeded the one-hour timeout."
            )

    finally:

        try:
            recognizer.stop_continuous_recognition_async().get()
        except Exception:
            pass

    if error_messages:
        raise RuntimeError("; ".join(error_messages))

    if not results:
        raise ValueError("No speech could be detected.")

    transcript = " ".join(
        item["text"] for item in results
    )

    words = [
        word
        for segment in results
        for word in segment["words"]
    ]

    duration = max(
        (
            segment["offset"] + segment["duration"]
            for segment in results
        ),
        default=0
    )

    available_confidences = [
        word["confidence"]
        for word in words
        if isinstance(word.get("confidence"), (int, float))
    ]

    average_confidence = (
        round(
            sum(available_confidences) /
            len(available_confidences),
            3
        )
        if available_confidences
        else None
    )

    return {
        "transcript": transcript,
        "language": detected_language,
        "duration_seconds": round(duration, 2),
        "confidence": average_confidence,
        "words": words,
        "segments": results
    }

# ============================================================
# AZURE LANGUAGE SERVICES
# ============================================================


def analyze_text(text):

    if not AZURE_LANGUAGE_KEY or not AZURE_LANGUAGE_ENDPOINT:
        raise RuntimeError("Azure Language credentials are missing.")

    if not text or not text.strip():
        raise ValueError("Cannot analyze empty text.")

    endpoint = AZURE_LANGUAGE_ENDPOINT.rstrip("/")

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY,
        "Content-Type": "application/json"
    }

    document = {
        "id": "1",
        "language": "en",
        "text": text[:5000]
    }

    analysis_url = (
        endpoint +
        "/language/:analyze-text?api-version=" +
        LANGUAGE_API_VERSION
    )

    def call_language_api(payload):

        response = requests.post(
            analysis_url,
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    # ------------------------------------------------
    # 1. SENTIMENT ANALYSIS
    # ------------------------------------------------

    sentiment_data = call_language_api({
        "kind": "SentimentAnalysis",
        "analysisInput": {
            "documents": [document]
        },
        "parameters": {
            "opinionMining": True
        }
    })

    sentiment_documents = sentiment_data.get(
        "results", {}
    ).get("documents", [])

    sentiment_result = {}

    if sentiment_documents:

        result = sentiment_documents[0]

        sentiment_result = {
            "sentiment": result.get("sentiment"),
            "confidence_scores": result.get(
                "confidenceScores", {}
            ),
            "sentences": result.get("sentences", [])
        }

    # ------------------------------------------------
    # 2. KEY PHRASE EXTRACTION
    # ------------------------------------------------

    keyphrase_data = call_language_api({
        "kind": "KeyPhraseExtraction",
        "analysisInput": {
            "documents": [document]
        },
        "parameters": {}
    })

    keyphrase_documents = keyphrase_data.get(
        "results", {}
    ).get("documents", [])

    key_phrases = []

    if keyphrase_documents:
        key_phrases = keyphrase_documents[0].get(
            "keyPhrases", []
        )

    # ------------------------------------------------
    # 3. NAMED ENTITY RECOGNITION
    # ------------------------------------------------

    ner_data = call_language_api({
        "kind": "EntityRecognition",
        "analysisInput": {
            "documents": [document]
        },
        "parameters": {
            "loggingOptOut": True
        }
    })

    ner_documents = ner_data.get(
        "results", {}
    ).get("documents", [])

    entities = []

    if ner_documents:

        for entity in ner_documents[0].get("entities", []):

            entities.append({
                "text": entity.get("text"),
                "category": entity.get("category"),
                "subcategory": entity.get("subcategory"),
                "confidence_score": entity.get("confidenceScore"),
                "offset": entity.get("offset"),
                "length": entity.get("length")
            })

    # Group entities into useful frontend categories.
    named_entities = {
        "people": [],
        "places": [],
        "organizations": [],
        "dates_and_times": [],
        "quantities": [],
        "other": []
    }

    category_mapping = {
        "Person": "people",
        "Location": "places",
        "Organization": "organizations",
        "DateTime": "dates_and_times",
        "Quantity": "quantities"
    }

    for entity in entities:

        category = entity.get("category")

        target = category_mapping.get(category, "other")

        named_entities[target].append(entity)

    return {
        "sentiment": sentiment_result,
        "key_phrases": key_phrases,
        "named_entities": named_entities,
        "entity_count": len(entities)
    }

# ============================================================
# MEETING MINUTES ANALYSIS
# ============================================================

def extract_meeting_details(text):
    """
    Extract meeting decisions, action items and scheduled meetings.

    Supports:
    - Decisions and agreements.
    - Individual action items and owners.
    - Due dates.
    - Scheduled meetings.
    - Dates with or without years.
    - Times such as 2 PM, 2:00 PM and 14:00.
    - Meeting dates and times separated by punctuation.
    - Prevents meetings from becoming action items.
    """

    decisions = []
    action_items = []
    meetings = []

    if not text or not text.strip():
        return {
            "decisions": [],
            "action_items": [],
            "action_item_count": 0,
            "meetings": []
        }

    current_year = datetime.now().year

    months = (
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December"
    )

    # ------------------------------------------------
    # DATE EXTRACTION
    # ------------------------------------------------

    def extract_date(value):

        if not value:
            return None

        match = re.search(
            rf"\b({months})\s+(\d{{1,2}})"
            rf"(?:st|nd|rd|th)?"
            rf"(?:,?\s+(\d{{4}}))?\b",
            value,
            re.IGNORECASE
        )

        if not match:
            return None

        month_name, day, year = match.groups()

        year = int(year) if year else current_year

        try:
            parsed = datetime.strptime(
                f"{month_name} {day} {year}",
                "%B %d %Y"
            )

            return parsed.strftime("%Y-%m-%d")

        except ValueError:
            return None

    # ------------------------------------------------
    # TIME EXTRACTION
    # ------------------------------------------------

    def normalize_time(value):

        if not value:
            return None

        value = value.strip().upper()

        formats = [
            "%I:%M %p",
            "%I %p",
            "%I:%M%p",
            "%I%p",
            "%H:%M"
        ]

        for fmt in formats:

            try:
                parsed = datetime.strptime(value, fmt)

                return parsed.strftime(
                    "%I:%M%p"
                ).lstrip("0")

            except ValueError:
                continue

        return None

    # ------------------------------------------------
    # NORMALIZE SENTENCES
    # ------------------------------------------------

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    normalized_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Join sentence fragments containing meeting
        # times or continuation phrases.

        if re.match(
            r'^(At|And|By|To)\s+',
            sentence,
            re.IGNORECASE
        ):

            if normalized_sentences:

                normalized_sentences[-1] += (
                    " " + sentence
                )

                continue

        normalized_sentences.append(sentence)

    # ------------------------------------------------
    # MEETING EXTRACTION
    # ------------------------------------------------

    meeting_patterns = [

        # Example:
        # We will meet again on November 23.
        # At 2:00 PM to review progress.

        rf"\b(?:we\s+)?(?:will\s+)?meet\s+"
        rf"(?:again\s+)?on\s+"
        rf"({months}\s+\d{{1,2}}"
        rf"(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?)"
        rf"[\s,.]*(?:at\s+)?"
        rf"(\d{{1,2}}(?::\d{{2}})?\s*(?:AM|PM))?",

        # Example:
        # The next meeting is scheduled for November 23 at 2 PM.

        rf"\b(?:next\s+)?meeting\s+(?:is\s+)?"
        rf"(?:scheduled\s+)?for\s+"
        rf"({months}\s+\d{{1,2}}"
        rf"(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?)"
        rf"[\s,.]*(?:at\s+)?"
        rf"(\d{{1,2}}(?::\d{{2}})?\s*(?:AM|PM))?",

        # Example:
        # We are meeting November 23 at 14:00.

        rf"\b(?:we\s+are\s+)?meeting\s+"
        rf"(?:again\s+)?on?\s*"
        rf"({months}\s+\d{{1,2}}"
        rf"(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?)"
        rf"[\s,.]*(?:at\s+)?"
        rf"(\d{{1,2}}(?::\d{{2}})?\s*(?:AM|PM))?"
    ]

    meeting_sentences = set()

    for sentence in normalized_sentences:

        for pattern in meeting_patterns:

            match = re.search(
                pattern,
                sentence,
                re.IGNORECASE
            )

            if not match:
                continue

            meeting_date = extract_date(
                match.group(1)
            )

            meeting_time = normalize_time(
                match.group(2)
            )

            if meeting_date:

                # Extract meeting description where possible.

                description_match = re.search(
                    r"(?:to|for)\s+(.+)",
                    sentence,
                    re.IGNORECASE
                )

                if description_match:

                    description = (
                        description_match.group(1)
                        .strip()
                        .rstrip(". ")
                    )

                else:

                    description = (
                        "Review everyone's progress."
                    )

                meetings.append({

                    "title": "Team Progress Review",

                    "date": meeting_date,

                    "time": meeting_time,

                    "description": description

                })

                meeting_sentences.add(sentence)

            break

    # ------------------------------------------------
    # DECISION EXTRACTION
    # ------------------------------------------------

    decision_patterns = [

        r"\bdecided to\b",

        r"\bdecided that\b",

        r"\bapproved\b",

        r"\bthe team agreed\b",

        r"\bthe team decided\b",

        r"\bwe agreed\b",

        r"\bit was agreed\b"

    ]

    for sentence in normalized_sentences:

        if any(
            re.search(
                pattern,
                sentence,
                re.IGNORECASE
            )
            for pattern in decision_patterns
        ):

            decisions.append({

                "description": sentence.rstrip(". ")

            })

    # ------------------------------------------------
    # ACTION ITEM EXTRACTION
    # ------------------------------------------------

    action_patterns = [

        r"\b([A-Z][a-z]+)\s+will\s+be\s+responsible\s+for\s+(.+)",

        r"\b([A-Z][a-z]+)\s+will\s+(.+)",

        r"\b([A-Z][a-z]+)\s+agreed\s+to\s+(.+)",

        r"\b([A-Z][a-z]+)\s+is\s+responsible\s+for\s+(.+)",

        r"\b([A-Z][a-z]+)\s+was\s+assigned\s+to\s+(.+)",

        r"\b([A-Z][a-z]+)\s+has\s+agreed\s+to\s+(.+)",

        r"\b([A-Z][a-z]+)\s+should\s+(.+)",

        r"\b([A-Z][a-z]+)\s+needs\s+to\s+(.+)"

    ]

    for sentence in normalized_sentences:

        # Exclude scheduled meeting sentences.

        if sentence in meeting_sentences:
            continue

        action_match = None

        for pattern in action_patterns:

            action_match = re.search(
                pattern,
                sentence,
                re.IGNORECASE
            )

            if action_match:
                break

        if not action_match:
            continue

        owner = action_match.group(1).strip()

        task = action_match.group(2).strip()

        task = task.rstrip(". ")

        # Extract due date.

        due_date = extract_date(sentence)

        # Remove due-date wording from the task.

        if due_date:

            task = re.sub(
                rf"\s+by\s+({months}\s+\d{{1,2}}"
                rf"(?:st|nd|rd|th)?"
                rf"(?:,?\s+\d{{4}})?)",
                "",
                task,
                flags=re.IGNORECASE
            ).strip()

        action_items.append({

            "task": task,

            "owner": owner,

            "due_date": due_date or "Not specified",

            "status": "pending"

        })

    # ------------------------------------------------
    # REMOVE DUPLICATE MEETINGS
    # ------------------------------------------------

    unique_meetings = []

    seen_meetings = set()

    for meeting in meetings:

        meeting_key = (

            meeting["date"],

            meeting["time"]

        )

        if meeting_key not in seen_meetings:

            seen_meetings.add(meeting_key)

            unique_meetings.append(meeting)

    # ------------------------------------------------
    # RETURN RESULTS
    # ------------------------------------------------

    return {

        "decisions": decisions,

        "action_items": action_items,

        "action_item_count": len(action_items),

        "meetings": unique_meetings

    }

# ============================================================
# MEETING MINUTES EXPORT GENERATION
# ============================================================

def escape_markdown(text):
    """Prevent special Markdown characters from breaking output."""
    return str(text or "").replace("\n", " ").strip()


def generate_markdown_export(decisions, action_items):
    """
    Generate a downloadable Markdown meeting minutes document.
    Includes decisions, action items, owners, due dates and status.
    """

    lines = [
        "# Meeting Minutes",
        "",
        f"**Generated:** {datetime.now().strftime('%B %d, %Y %H:%M')}",
        "",
        "---",
        "",
        "## Decisions",
        ""
    ]

    if decisions:
        for index, decision in enumerate(decisions, 1):
            description = escape_markdown(
                decision.get("description", "")
            )
            lines.append(f"{index}. {description}")
    else:
        lines.append("No decisions identified.")

    lines.extend([
        "",
        "## Action Items",
        ""
    ])

    if action_items:
        lines.extend([
            "| # | Task | Owner | Due Date | Status |",
            "|---|---|---|---|---|"
        ])

        for index, item in enumerate(action_items, 1):
            task = escape_markdown(item.get("task", ""))
            owner = escape_markdown(item.get("owner", "Unassigned"))
            due_date = escape_markdown(
                item.get("due_date", "Not specified")
            )
            status = escape_markdown(
                item.get("status", "pending")
            )

            # Escape Markdown table separators.
            task = task.replace("|", r"\|")
            owner = owner.replace("|", r"\|")
            due_date = due_date.replace("|", r"\|")
            status = status.replace("|", r"\|")

            lines.append(
                f"| {index} | {task} | {owner} | "
                f"{due_date} | {status} |"
            )

    else:
        lines.append("No action items identified.")

    lines.extend([
        "",
        "---",
        "",
        f"**Total Decisions:** {len(decisions)}",
        f"**Total Action Items:** {len(action_items)}",
        "",
        "Generated by Meeting Minutes Bot."
    ])

    return "\n".join(lines)


def escape_ics(text):
    """Escape text according to iCalendar text rules."""

    text = str(text or "")

    return (
        text.replace("\\", "\\\\")
        .replace(";", r"\;")
        .replace(",", r"\,")
        .replace("\r\n", r"\n")
        .replace("\n", r"\n")
        .replace("\r", r"\n")
    )


def parse_ics_due_date(value):
    """
    Parse supported due-date formats.

    Supported:
    - YYYY-MM-DD
    - January 15
    - January 15, 2027

    Dates without a year use the current year.
    """

    if not value or value == "Not specified":
        return None

    value = str(value).strip()

    formats = [
        "%Y-%m-%d",
        "%B %d, %Y",
        "%B %d"
    ]

    for date_format in formats:
        try:
            parsed = datetime.strptime(value, date_format)

            if date_format == "%B %d":
                parsed = parsed.replace(year=datetime.now().year)

            return parsed.date()

        except ValueError:
            continue

    return None




def generate_ics_export(decisions, action_items, meetings=None):
    """
    Generate a Google Calendar-compatible ICS file.

    Exports:
    - Action item deadlines as all-day VEVENT entries.
    - Scheduled meetings as timed VEVENT entries.
    - Meetings without times as all-day events.

    All entries use VEVENT for compatibility with
    standard calendar event imports.
    """

    meetings = meetings or []
    decisions = decisions or []
    action_items = action_items or []

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Meeting Minutes Bot//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:Meeting Minutes"
    ]

    created = datetime.utcnow().strftime(
        "%Y%m%dT%H%M%SZ"
    )

    # ------------------------------------------------
    # DECISION CONTEXT
    # ------------------------------------------------

    decision_text = "\n".join(
        decision.get("description", "")
        for decision in decisions
        if isinstance(decision, dict)
    )

    # ------------------------------------------------
    # ACTION ITEMS AS ALL-DAY EVENTS
    # ------------------------------------------------

    for item in action_items:

        task = item.get(
            "task",
            "Untitled task"
        )

        owner = item.get(
            "owner",
            "Unassigned"
        )

        due_date = item.get(
            "due_date",
            "Not specified"
        )

        status = item.get(
            "status",
            "pending"
        )

        parsed_date = parse_ics_due_date(
            due_date
        )

        if not parsed_date:
            continue

        next_day = parsed_date + timedelta(days=1)

        uid = (
            str(uuid.uuid4()) +
            "@meetingminutesbot"
        )

        description = (
            f"Owner: {owner}\n"
            f"Status: {status}\n"
            f"Decision context: {decision_text}"
        )

        lines.extend([

            "BEGIN:VEVENT",

            f"UID:{uid}",

            f"DTSTAMP:{created}",

            f"DTSTART;VALUE=DATE:{parsed_date.strftime('%Y%m%d')}",

            f"DTEND;VALUE=DATE:{next_day.strftime('%Y%m%d')}",

            f"SUMMARY:{escape_ics(task)}",

            f"DESCRIPTION:{escape_ics(description)}",

            "CATEGORIES:Action Item",

            "END:VEVENT"

        ])

    # ------------------------------------------------
    # MEETINGS AS CALENDAR EVENTS
    # ------------------------------------------------

    for meeting in meetings:

        meeting_date = meeting.get("date")

        meeting_time = meeting.get("time")

        if not meeting_date:
            continue

        try:

            parsed_date = datetime.strptime(
                meeting_date,
                "%Y-%m-%d"
            )

        except ValueError:
            continue

        uid = (
            str(uuid.uuid4()) +
            "@meetingminutesbot"
        )

        title = meeting.get(
            "title",
            "Team Meeting"
        )

        description = meeting.get(
            "description",
            "Scheduled meeting."
        )

        # ------------------------------------------------
        # ALL-DAY MEETING
        # ------------------------------------------------

        if not meeting_time:

            next_day = parsed_date + timedelta(days=1)

            lines.extend([

                "BEGIN:VEVENT",

                f"UID:{uid}",

                f"DTSTAMP:{created}",

                f"DTSTART;VALUE=DATE:{parsed_date.strftime('%Y%m%d')}",

                f"DTEND;VALUE=DATE:{next_day.strftime('%Y%m%d')}",

                f"SUMMARY:{escape_ics(title)}",

                f"DESCRIPTION:{escape_ics(description)}",

                "CATEGORIES:Meeting",

                "END:VEVENT"

            ])

            continue

        # ------------------------------------------------
        # TIMED MEETING
        # ------------------------------------------------

        parsed_datetime = parse_meeting_datetime(
            meeting_date,
            meeting_time
        )

        if not parsed_datetime:
            continue

        end_datetime = (
            parsed_datetime + timedelta(hours=1)
        )

        start = parsed_datetime.strftime(
            "%Y%m%dT%H%M%S"
        )

        end = end_datetime.strftime(
            "%Y%m%dT%H%M%S"
        )

        lines.extend([

            "BEGIN:VEVENT",

            f"UID:{uid}",

            f"DTSTAMP:{created}",

            f"DTSTART:{start}",

            f"DTEND:{end}",

            f"SUMMARY:{escape_ics(title)}",

            f"DESCRIPTION:{escape_ics(description)}",

            "CATEGORIES:Meeting",

            "END:VEVENT"

        ])

    # ------------------------------------------------
    # CLOSE CALENDAR
    # ------------------------------------------------

    lines.append("END:VCALENDAR")

    return "\r\n".join(lines) + "\r\n"

# ============================================================
# MEETING MINUTES EXPORT API
# ============================================================

def get_export_data():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        raise ValueError("A JSON request body is required.")

    decisions = data.get("decisions", [])
    action_items = data.get("action_items", [])
    meetings = data.get("meetings", [])

    if not all(isinstance(x, list) for x in [
        decisions, action_items, meetings
    ]):
        raise ValueError("Decisions, action items and meetings must be lists.")

    for collection in [decisions, action_items, meetings]:
        if not all(isinstance(item, dict) for item in collection):
            raise ValueError("Each item must be a JSON object.")

    return decisions, action_items, meetings

@app.route("/export/markdown", methods=["POST"])
def export_markdown():

    try:
        decisions, action_items, meetings = get_export_data()

        markdown_content = generate_markdown_export(
            decisions,
            action_items
        )

        file_data = BytesIO(
            markdown_content.encode("utf-8")
        )

        return send_file(
            file_data,
            mimetype="text/markdown",
            as_attachment=True,
            download_name="meeting_minutes.md"
        )

    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception as error:
        app.logger.exception("Markdown export failed.")

        return jsonify({
            "success": False,
            "error": "Could not generate Markdown export."
        }), 500


@app.route("/export/ics", methods=["POST"])
def export_ics():

    try:
        decisions, action_items, meetings = get_export_data()

        ics_content = generate_ics_export(
            decisions,
            action_items,
            meetings
        )

        file_data = BytesIO(ics_content.encode("utf-8"))

        return send_file(
            file_data,
            mimetype="text/calendar; charset=utf-8",
            as_attachment=True,
            download_name="meeting_minutes.ics"
        )

    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("Calendar export failed.")

        return jsonify({
            "success": False,
            "error": "Could not generate calendar export."
        }), 500

def parse_meeting_datetime(meeting_date, meeting_time):

    if not meeting_date or not meeting_time:
        return None

    formats = [
        "%Y-%m-%d %I:%M%p",
        "%Y-%m-%d %I%p",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I %p"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(
                f"{meeting_date} {meeting_time.strip().upper()}",
                fmt
            )
        except ValueError:
            continue

    return None


def process_audio_upload(uploaded_file):

    if uploaded_file is None or not uploaded_file.filename:
        raise ValueError("Please upload an audio file.")

    original_filename = uploaded_file.filename

    # Extract and validate extension
    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported audio format. "
            "Supported formats: WAV, MP3, WebM, OGG, "
            "Opus, AAC and M4A."
        )

    input_path = None
    wav_path = None

    try:

        # -----------------------------------------
        # STEP 1: SAVE ORIGINAL AUDIO
        # -----------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_input:

            input_path = temp_input.name
            uploaded_file.save(input_path)

        if os.path.getsize(input_path) == 0:
            raise ValueError("The uploaded audio file is empty.")

        # -----------------------------------------
        # STEP 2: CREATE TEMPORARY WAV FILE
        # -----------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_wav:

            wav_path = temp_wav.name

        # -----------------------------------------
        # STEP 3: CONVERT ALL FORMATS TO WAV
        # -----------------------------------------

        convert_to_wav(
            input_path,
            wav_path
        )

        # -----------------------------------------
        # STEP 4: TRANSCRIBE USING AZURE SPEECH
        # -----------------------------------------

        transcription = transcribe_audio(wav_path)

        return {
            "filename": original_filename,
            "original_format": extension,
            "converted_format": "wav",
            "sample_rate": 16000,
            "channels": 1,
            "encoding": "PCM signed 16-bit",
            **transcription
        }

    finally:

        # -----------------------------------------
        # STEP 5: CLEAN UP TEMPORARY FILES
        # -----------------------------------------

        for path in (input_path, wav_path):

            if path and os.path.exists(path):

                try:
                    os.remove(path)

                except OSError:
                    app.logger.warning(
                        "Could not delete temporary file: %s",
                        path
                    )

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json(silent=True)

    if not data or not isinstance(data.get("text"), str):
        return jsonify({
            "success": False,
            "error": "Please provide a text field containing the transcript."
        }), 400

    text = clean_transcript(data["text"])

    if not text:
        return jsonify({
            "success": False,
            "error": "Transcript cannot be empty."
        }), 400

    try:

        # Azure Language analysis
        language_results = analyze_text(text)

        # Meeting decisions and action items
        meeting_results = extract_meeting_details(text)

        return jsonify({

            "success": True,

            "transcript": text,

            "meeting_summary": {

                "sentiment": language_results["sentiment"].get(
                    "sentiment"
                ),

                "confidence_scores": language_results["sentiment"].get(
                    "confidence_scores", {}
                ),

                "key_phrases": language_results["key_phrases"]
            },

            "named_entities": language_results["named_entities"],

            "entity_count": language_results["entity_count"],

            "decisions": meeting_results["decisions"],

            "action_items": meeting_results["action_items"],

            "action_item_count": meeting_results["action_item_count"],
            
            "meetings": meeting_results["meetings"]

        }), 200

    except requests.exceptions.Timeout:

        return jsonify({
            "success": False,
            "error": "Azure Language service timed out."
        }), 504

    except requests.exceptions.HTTPError as error:

        return jsonify({
            "success": False,
            "error": "Azure Language API returned an HTTP error.",
            "details": str(error)
        }), 502

    except requests.exceptions.RequestException as error:

        return jsonify({
            "success": False,
            "error": "Azure Language API request failed.",
            "details": str(error)
        }), 502

    except Exception as error:

        app.logger.exception("Meeting analysis failed.")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


@app.route("/process", methods=["POST"])
def process():

    try:

        # ---------------------------------------------
        # STEP 1: RECEIVE AUDIO
        # ---------------------------------------------

        uploaded_file = request.files.get("file")

        # ---------------------------------------------
        # STEP 2: TRANSCRIBE AUDIO
        # ---------------------------------------------

        transcription = process_audio_upload(uploaded_file)

        transcript = clean_transcript(
            transcription["transcript"]
        )

        # ---------------------------------------------
        # STEP 3: AZURE LANGUAGE ANALYSIS
        # ---------------------------------------------

        language_results = analyze_text(transcript)

        # ---------------------------------------------
        # STEP 4: MEETING ANALYSIS
        # ---------------------------------------------

        meeting_results = extract_meeting_details(transcript)

        # ---------------------------------------------
        # STEP 5: RETURN COMPLETE RESULTS
        # ---------------------------------------------

        return jsonify({

            "success": True,

            "file": {
                "filename": transcription["filename"]
            },

            # PART B: SPEECH TRANSCRIPTION
            "transcription": {

                "full_text": transcript,

                "language": transcription["language"],

                "duration_seconds": transcription["duration_seconds"],

                "confidence": transcription["confidence"],

                "words": transcription["words"]
            },

            # PART C1: AZURE LANGUAGE ANALYSIS
            "analysis": {

                "sentiment": language_results["sentiment"].get(
                    "sentiment"
                ),

                "confidence_scores": language_results["sentiment"].get(
                    "confidence_scores", {}
                ),

                "key_phrases": language_results["key_phrases"],

                "named_entities": language_results["named_entities"],

                "entity_count": language_results["entity_count"]

            },

            # MEETING MINUTES
            "meeting": {

                "decisions": meeting_results["decisions"],

                "action_items": meeting_results["action_items"],

                "action_item_count": meeting_results["action_item_count"],
                
                "meetings": meeting_results["meetings"]
            }

        }), 200

    except ValueError as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except TimeoutError as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 504

    except requests.exceptions.Timeout:

        return jsonify({
            "success": False,
            "error": "Azure Language service timed out."
        }), 504

    except requests.exceptions.HTTPError as error:

        return jsonify({
            "success": False,
            "error": "Azure Language API returned an HTTP error.",
            "details": str(error)
        }), 502

    except requests.exceptions.RequestException as error:

        return jsonify({
            "success": False,
            "error": "Azure Language API request failed.",
            "details": str(error)
        }), 502

    except Exception as error:

        app.logger.exception(
            "Complete processing pipeline failed."
        )

        return jsonify({
            "success": False,
            "error": "Audio processing pipeline failed.",
            "details": str(error)
        }), 500

# ============================================================
# TRANSCRIPT CLEANUP
# ============================================================

def clean_transcript(text):
    """
    Clean unnecessary whitespace and formatting
    from the generated transcript.
    """

    if not text:
        return ""

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# FLASK API ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "application": "Meeting Transcription API",
        "message": "Azure Speech and Language services are available."
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "speech_configured": bool(
            AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
        ),
        "language_configured": bool(
            AZURE_LANGUAGE_KEY and AZURE_LANGUAGE_ENDPOINT
        )
    })


# ============================================================
# AUDIO UPLOAD AND PROCESSING
# ============================================================

@app.route("/transcribe", methods=["POST"])
def transcribe():

    try:

        uploaded_file = request.files.get("file")

        transcription = process_audio_upload(uploaded_file)

        return jsonify({
            "success": True,
            "filename": transcription["filename"],
            "transcript": transcription["transcript"],
            "language": transcription["language"],
            "duration_seconds": transcription["duration_seconds"],
            "confidence": transcription["confidence"],
            "words": transcription["words"]
        }), 200

    except ValueError as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except TimeoutError as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 504

    except Exception as error:

        app.logger.exception("Transcription failed.")

        return jsonify({
            "success": False,
            "error": "Audio transcription failed.",
            "details": str(error)
        }), 500

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "File is too large. Maximum upload size is 25 MB."
    }), 413


@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "success": False,
        "error": "The requested endpoint does not exist."
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):

    return jsonify({
        "success": False,
        "error": "This HTTP method is not allowed for this endpoint."
    }), 405

# ============================================================
# APPLICATION STARTUP
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=False,
        threaded=True
    )