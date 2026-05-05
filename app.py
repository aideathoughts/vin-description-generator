import datetime as dt
import io
import json
import os
import sqlite3
from dataclasses import dataclass

import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


DB_PATH = "vehicles.db"
DEALER_NAME = "Galaxy Automotive LLC"
DEALER_ADDRESS = "12719 N Florida Ave, Tampa"
DEALER_PHONE = "8133541236"


def resolve_api_key(override: str = "") -> str:
    key = (override or "").strip()
    if key:
        return key

    env_key = (os.getenv("OPENAI_API_KEY", "") or "").strip()
    if env_key:
        return env_key

    try:
        secret_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        secret_key = ""
    return secret_key


@dataclass
class VehicleInput:
    vin: str
    mileage: str
    down_payment: str
    color: str
    price: str
    title_status: str
    condition_note: str
    condition_report_text: str = ""
    year: str = ""
    make: str = ""
    model: str = ""
    trim: str = ""
    body_style: str = ""
    engine: str = ""

    @property
    def title(self) -> str:
        parts = [self.year, self.make, self.model, self.trim]
        clean = " ".join(p for p in parts if p and p.strip()).strip()
        if clean:
            return f"{clean} - Available Now"
        return "Vehicle - Available Now"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vin TEXT,
            year TEXT,
            make TEXT,
            model TEXT,
            trim TEXT,
            body_style TEXT,
            engine TEXT,
            mileage TEXT,
            down_payment TEXT,
            color TEXT,
            price TEXT,
            title_status TEXT,
            condition_note TEXT,
            generated_description TEXT,
            edited_description TEXT,
            status TEXT,
            date_created TEXT
        )
        """
    )
    _ensure_column(cur, "vehicles", "down_payment", "TEXT")
    conn.commit()
    conn.close()


def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, column_type: str) -> None:
    cursor.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if column not in existing_columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def decode_vin(vin: str) -> dict:
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json().get("Results", [{}])[0]
    return {
        "year": data.get("ModelYear", "") or "",
        "make": data.get("Make", "") or "",
        "model": data.get("Model", "") or "",
        "trim": data.get("Trim", "") or "",
        "body_style": data.get("BodyClass", "") or "",
        "engine": _get_engine_string(data),
    }


def _get_engine_string(vpic_row: dict) -> str:
    displacement = (vpic_row.get("DisplacementL", "") or "").strip()
    cylinders = (vpic_row.get("EngineCylinders", "") or "").strip()
    fuel = (vpic_row.get("FuelTypePrimary", "") or "").strip()

    pieces = []
    if displacement:
        pieces.append(f"{displacement}L")
    if cylinders:
        pieces.append(f"{cylinders}-cyl")
    if fuel:
        pieces.append(fuel)
    return " ".join(pieces).strip()


def _vehicle_data_block(vehicle: VehicleInput) -> str:
    vehicle_data_lines = [
        f"Year: {vehicle.year}",
        f"Make: {vehicle.make}",
        f"Model: {vehicle.model}",
        f"Trim: {vehicle.trim}",
        f"Mileage: {vehicle.mileage}",
        f"Body Style: {vehicle.body_style}",
        f"Engine: {vehicle.engine}",
        f"Dealer: {DEALER_NAME}",
        f"Address: {DEALER_ADDRESS}",
        f"Dealer Number: {DEALER_PHONE}",
    ]

    optional_lines = [
        ("Color", vehicle.color),
        ("Down Payment", vehicle.down_payment),
        ("Price", vehicle.price),
        ("Title Status", vehicle.title_status),
        ("Condition Note", vehicle.condition_note),
    ]
    for label, value in optional_lines:
        if value and value.strip():
            vehicle_data_lines.append(f"{label}: {value.strip()}")
    if vehicle.condition_report_text and vehicle.condition_report_text.strip():
        vehicle_data_lines.append("Condition Report Text:")
        vehicle_data_lines.append(vehicle.condition_report_text.strip()[:3500])
    return "\n".join(vehicle_data_lines)


def build_generation_prompt(
    vehicle: VehicleInput, variation_token: str = "", previous_description: str = ""
) -> str:
    vehicle_data_block = _vehicle_data_block(vehicle)
    previous_description_block = previous_description.strip()[:900]
    return f"""Generate a clean car sales description for Facebook Marketplace and dealership use.
Do not use a "key features" section.
Do not make claims that are not provided.
Do not say "perfect condition."
Do not overpromise.
Keep it natural, simple, and professional.
Make it sound trustworthy and buyer-friendly.
Use varied wording and avoid repeating the same sentence structure.
Keep it around 120-190 words.
Mention that buy here pay here financing options are available.
Include a short pricing options section that mentions price and down payment when provided.
Do not mention price, down payment, or any payment amount unless those values are explicitly provided.
If no price/down payment is provided, omit any pricing section completely.
Do not include placeholders like "N/A", "unknown", or blank labels.
If a field value is missing, omit that field entirely.
When condition note is provided, naturally include it in the description.
When condition report text is provided, use it to improve accuracy and sales wording.
Use condition report details as supporting facts, but keep the final output concise and buyer-friendly.
If VIN, mileage, or other fields are missing in form inputs, infer them from condition report text when clearly available.
Create a fresh variation each time this prompt is run.
Variation token: {variation_token}
If there is a previous description, write a meaningfully different version with new phrasing and structure.

Previous description (for variation only, do not copy verbatim):
{previous_description_block}

Vehicle data:
{vehicle_data_block}
"""


def build_paraphrase_prompt(vehicle: VehicleInput, source_text: str, variation_token: str = "") -> str:
    vehicle_data_block = _vehicle_data_block(vehicle)
    return f"""Paraphrase the text below into a clean, buyer-friendly vehicle sales description.
Keep the meaning accurate.
Do not add claims that are not supported by the provided data.
Do not use a "key features" section.
Do not use placeholders like N/A.
When condition note is provided, naturally include it.
When condition report text is provided, use it to preserve key factual details while improving wording.
Keep it around 120-190 words.
Variation token: {variation_token}

Vehicle data:
{vehicle_data_block}

Text to paraphrase:
{source_text}
"""


def generate_description(
    vehicle: VehicleInput,
    api_key_override: str = "",
    generation_mode: str = "new",
    paraphrase_text: str = "",
    previous_description: str = "",
) -> tuple[str, str]:
    api_key = resolve_api_key(api_key_override)
    variation_token = dt.datetime.now().isoformat(timespec="microseconds")
    mode = generation_mode.strip().lower()
    if mode == "paraphrase":
        if OpenAI is None:
            raise RuntimeError("OpenAI package is not available in this runtime. Reinstall dependencies and restart.")
        if not api_key:
            raise RuntimeError("No OpenAI API key detected in environment/secrets.")
    if api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            if mode == "paraphrase":
                if not paraphrase_text.strip():
                    raise RuntimeError("Paraphrase mode needs source text.")
                prompt = build_paraphrase_prompt(vehicle, paraphrase_text, variation_token)
                temperature = 0.8
            else:
                prompt = build_generation_prompt(vehicle, variation_token, previous_description)
                temperature = 1.0
            result = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
                temperature=temperature,
            )
            text = (result.output_text or "").strip()
            if text:
                return text, "OpenAI"
            raise RuntimeError("OpenAI returned empty output.")
        except Exception as exc:
            error_text = str(exc)
            error_lower = error_text.lower()
            if "insufficient_quota" in error_lower:
                raise RuntimeError(
                    "OpenAI key is set, but quota/billing is exhausted. Add credits or use a funded key."
                ) from exc
            if "invalid_api_key" in error_lower:
                raise RuntimeError("OpenAI API key is invalid. Paste a valid key in the sidebar.") from exc
            raise RuntimeError(f"OpenAI generation failed: {error_text}") from exc
    if mode == "paraphrase":
        raise RuntimeError("Paraphrase mode requires a valid OpenAI API key.")
    return fallback_description(vehicle), "Template"


def test_openai_connection(api_key: str) -> tuple[bool, str]:
    key = resolve_api_key(api_key)
    if not key:
        return False, "No API key entered."
    if OpenAI is None:
        return False, "OpenAI package is not installed in this runtime."
    try:
        client = OpenAI(api_key=key)
        result = client.responses.create(
            model="gpt-4.1-mini",
            input="Reply with only OK",
            temperature=0,
        )
        text = (result.output_text or "").strip().upper()
        if "OK" in text:
            return True, "OpenAI key is working."
        return True, "OpenAI responded, but with unexpected output."
    except Exception as exc:
        return False, f"OpenAI test failed: {exc}"


def fallback_description(vehicle: VehicleInput) -> str:
    vehicle_name = " ".join(
        p for p in [vehicle.year, vehicle.make, vehicle.model, vehicle.trim] if p and p.strip()
    ).strip()
    vehicle_name = vehicle_name or "vehicle"

    color_prefix = f"{vehicle.color.strip()} " if vehicle.color and vehicle.color.strip() else ""
    description_lines = [
        f"{vehicle_name} - Available Now",
        (
            f"This {color_prefix}{vehicle_name} is available with {vehicle.mileage} miles. "
            "It offers a clean look, comfortable drive, and practical setup for daily use."
        ),
        (
            f"The vehicle is available now at {DEALER_NAME}, {DEALER_ADDRESS}. "
            "Buy here pay here financing options may be available. "
            f"Message us today at {DEALER_PHONE} to check availability, ask questions, or schedule a test drive."
        ),
    ]

    if vehicle.mileage:
        description_lines.append(f"Mileage: {vehicle.mileage}")
    if vehicle.color:
        description_lines.append(f"Color: {vehicle.color}")
    if vehicle.title_status:
        description_lines.append(f"Title: {vehicle.title_status}")
    if vehicle.condition_note:
        description_lines.append(f"Condition Note: {vehicle.condition_note}")
    if vehicle.condition_report_text:
        report_highlight = " ".join(vehicle.condition_report_text.split())[:320]
        description_lines.append(f"Condition Report: {report_highlight}")

    pricing_lines = []
    if vehicle.price:
        pricing_lines.append(f"Price: {vehicle.price}")
    if vehicle.down_payment:
        pricing_lines.append(f"Down Payment: {vehicle.down_payment}")
    if pricing_lines:
        description_lines.append("Pricing Options:")
        description_lines.extend(pricing_lines)

    return "\n".join(description_lines)


def save_vehicle(vehicle: VehicleInput, generated_description: str, edited_description: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO vehicles (
            vin, year, make, model, trim, body_style, engine,
            mileage, down_payment, color, price, title_status, condition_note,
            generated_description, edited_description, status, date_created
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            vehicle.vin,
            vehicle.year,
            vehicle.make,
            vehicle.model,
            vehicle.trim,
            vehicle.body_style,
            vehicle.engine,
            vehicle.mileage,
            vehicle.down_payment,
            vehicle.color,
            vehicle.price,
            vehicle.title_status,
            vehicle.condition_note,
            generated_description,
            edited_description,
            "draft",
            dt.datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_rows(limit: int = 20) -> list[sqlite3.Row]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT vin, year, make, model, trim, mileage, price, status, date_created
        FROM vehicles
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def _init_session_state() -> None:
    defaults = {
        "vehicle": None,
        "generated_title": "",
        "generated_description": "",
        "editable_description": "",
        "copy_payload": "",
        "openai_api_key": "",
        "vin_input": "",
        "mileage_input": "",
        "down_payment_input": "",
        "color_input": "",
        "price_input": "",
        "title_status_input": "",
        "condition_note_input": "",
        "condition_report_text": "",
        "condition_report_parse_message": "",
        "last_generation_source": "",
        "last_generation_error": "",
        "generation_mode_input": "New Text Generation",
        "paraphrase_source_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _extract_condition_report_text(uploaded_file) -> tuple[str, str]:
    if uploaded_file is None:
        return "", ""

    try:
        raw_bytes = uploaded_file.getvalue()
    except Exception as exc:
        return "", f"Could not read uploaded file: {exc}"

    if not raw_bytes:
        return "", "Condition report file is empty."

    file_name = (getattr(uploaded_file, "name", "") or "").lower()
    if file_name.endswith(".pdf"):
        if PdfReader is None:
            return "", "PDF parsing unavailable. Install `pypdf` to read PDF reports."
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = []
            for page in reader.pages:
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages.append(page_text)
            full_text = "\n".join(pages).strip()
            if not full_text:
                return "", "PDF uploaded, but no readable text was found."
            return full_text[:7000], ""
        except Exception as exc:
            return "", f"Failed to parse PDF condition report: {exc}"

    decoded_text = ""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            decoded_text = raw_bytes.decode(encoding)
            break
        except Exception:
            continue
    if not decoded_text:
        return "", "Could not decode file text. Please upload a PDF or plain text file."

    cleaned_text = decoded_text.replace("\x00", " ").strip()
    if not cleaned_text:
        return "", "Uploaded text file has no readable content."
    return cleaned_text[:7000], ""


def _build_vehicle_from_inputs() -> VehicleInput:
    vin = st.session_state.get("vin_input", "").strip().upper()
    mileage = st.session_state.get("mileage_input", "").strip()
    down_payment = st.session_state.get("down_payment_input", "").strip()
    color = st.session_state.get("color_input", "").strip()
    price = st.session_state.get("price_input", "").strip()
    title_status = st.session_state.get("title_status_input", "").strip()
    condition_note = st.session_state.get("condition_note_input", "").strip()
    condition_report_file = st.session_state.get("condition_report_file")
    condition_report_text, report_message = _extract_condition_report_text(condition_report_file)
    st.session_state["condition_report_text"] = condition_report_text
    st.session_state["condition_report_parse_message"] = report_message

    decoded = {}
    if vin:
        decoded = decode_vin(vin)
    return VehicleInput(
        vin=vin,
        mileage=mileage,
        down_payment=down_payment,
        color=color,
        price=price,
        title_status=title_status,
        condition_note=condition_note,
        condition_report_text=condition_report_text,
        year=decoded.get("year", ""),
        make=decoded.get("make", ""),
        model=decoded.get("model", ""),
        trim=decoded.get("trim", ""),
        body_style=decoded.get("body_style", ""),
        engine=decoded.get("engine", ""),
    )


def _generate_and_store_from_current_inputs(generation_mode: str, paraphrase_text: str = "") -> None:
    vehicle = _build_vehicle_from_inputs()
    previous_description = st.session_state.get("generated_description", "")
    generated_description, source = generate_description(
        vehicle,
        "",
        generation_mode=generation_mode,
        paraphrase_text=paraphrase_text,
        previous_description=previous_description,
    )
    st.session_state["vehicle"] = vehicle
    st.session_state["generated_title"] = vehicle.title
    st.session_state["generated_description"] = generated_description
    st.session_state["editable_description"] = generated_description
    st.session_state["copy_payload"] = f"{vehicle.title}\n{generated_description}"
    st.session_state["last_generation_source"] = source
    st.session_state["last_generation_error"] = ""


def render_editor_page() -> None:
    st.subheader("VIN Description Editor")
    st.caption(
        "Use one click to generate or paraphrase. Output appears right below the form. "
        "You can generate from condition report + pricing only."
    )

    with st.form("vehicle_form"):
        st.text_input("VIN", max_chars=17, key="vin_input")
        st.text_input("Mileage", placeholder="82,400", key="mileage_input")
        st.selectbox(
            "Generation Mode",
            ["New Text Generation", "Paraphrase Existing Text"],
            key="generation_mode_input",
        )
        if st.session_state.get("generation_mode_input") == "Paraphrase Existing Text":
            st.text_area(
                "Text to Paraphrase (optional - leave blank to use current editable text)",
                key="paraphrase_source_input",
                placeholder="Paste text to rewrite...",
                height=110,
            )
        with st.expander("Optional fields", expanded=False):
            st.text_input("Down Payment", placeholder="$2,000", key="down_payment_input")
            st.text_input("Color", placeholder="Silver", key="color_input")
            st.text_input("Price", placeholder="$14,500", key="price_input")
            st.text_input("Title Status", placeholder="Clean Title", key="title_status_input")
            st.text_area("Condition Note", placeholder="Runs and drives well.", key="condition_note_input")
            st.file_uploader(
                "Condition Report File (PDF or text)",
                key="condition_report_file",
                help="Attach a PDF or text file. The model will read it and use it to improve the description.",
            )
        submitted = st.form_submit_button("Generate Description")

    if submitted:
        vin = st.session_state.get("vin_input", "").strip()
        mileage = st.session_state.get("mileage_input", "").strip()
        condition_report_file = st.session_state.get("condition_report_file")
        has_condition_report = condition_report_file is not None
        if not vin and not has_condition_report:
            st.error("Enter VIN or upload a condition report file.")
            return
        if not mileage and not has_condition_report:
            st.error("Enter mileage or upload a condition report file that includes mileage.")
            return

        selected_mode = st.session_state.get("generation_mode_input", "New Text Generation")
        generation_mode = "paraphrase" if selected_mode == "Paraphrase Existing Text" else "new"
        paraphrase_text = st.session_state.get("paraphrase_source_input", "").strip()
        if generation_mode == "paraphrase" and not paraphrase_text:
            paraphrase_text = st.session_state.get("editable_description", "").strip() or st.session_state.get(
                "generated_description", ""
            ).strip()

        try:
            with st.spinner("Decoding VIN and generating description..."):
                _generate_and_store_from_current_inputs(generation_mode, paraphrase_text)
            st.success("Description generated.")
            if st.session_state.get("condition_report_text"):
                st.caption("Condition report text loaded and included in generation.")
            if st.session_state.get("condition_report_parse_message"):
                st.warning(st.session_state.get("condition_report_parse_message"))
            st.caption("VIN data is sourced from NHTSA vPIC and should be verified before posting.")
        except Exception as exc:
            st.session_state["last_generation_error"] = str(exc)
            st.error(str(exc))
            return

    vehicle = st.session_state.get("vehicle")
    if not vehicle:
        return

    st.divider()
    st.markdown(f"**Generated Title:** {st.session_state.get('generated_title', '')}")
    st.caption(f"Generated with: {st.session_state.get('last_generation_source', 'Unknown')}")
    st.markdown("**Generated Description:**")
    st.write(st.session_state.get("generated_description", ""))

    edited = st.text_area(
        "Editable Text Box",
        value=st.session_state.get("editable_description", ""),
        height=260,
    )
    st.session_state["editable_description"] = edited
    st.session_state["copy_payload"] = f"{st.session_state.get('generated_title', '')}\n{edited}"

    c1, c2 = st.columns(2)
    if c1.button("Save"):
        save_vehicle(
            st.session_state.get("vehicle"),
            st.session_state.get("generated_description", ""),
            st.session_state.get("editable_description", ""),
        )
        st.success("Saved to local database.")

    copy_payload = st.session_state.get("copy_payload", "")
    with c2:
        components.html(
            f"""
            <button onclick='navigator.clipboard.writeText({json.dumps(copy_payload)}); this.innerText="Copied!";'
                style="padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid #999; cursor: pointer;">
                Copy
            </button>
            """,
            height=45,
        )


def render_recent_vehicles() -> None:
    st.subheader("Recent Saved Vehicles")
    rows = get_recent_rows()
    if not rows:
        st.caption("No saved vehicles yet.")
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="VIN Description Editor", layout="wide")
    init_db()
    _init_session_state()

    st.title("VIN-Based Description Editor")
    st.caption(
        "Flow: VIN + miles (+ optional details) -> decode VIN -> generate -> edit -> save/copy."
    )

    page = st.sidebar.radio("Navigation", ["Editor", "Saved Vehicles"])
    if page == "Editor":
        render_editor_page()
    else:
        render_recent_vehicles()


if __name__ == "__main__":
    main()
