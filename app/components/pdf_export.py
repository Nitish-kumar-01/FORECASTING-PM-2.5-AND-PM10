"""
PDF Report Generation Module for Korba Air Quality Forecasts.
Implements FR-13 using ReportLab.
"""

from io import BytesIO
from typing import List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.aqi import DailyAQIResult
from src.precautions import DailyAdvisory, HourlySpikeAlert


def generate_forecast_pdf(
    station_name: str,
    daily_results: List[DailyAQIResult],
    advisories: List[DailyAdvisory],
    spike_alert: Optional[HourlySpikeAlert] = None,
    issue_date: str = "2026-11-19 12:00 IST",
    model_version: str = "Multi-Station LSTM v1.0",
) -> bytes:
    """
    Generates a printable PDF bulletin summarizing 72-hour air quality predictions,
    AQI categories, health precautions, and severe spike warnings.
    Returns:
        PDF file bytes suitable for Streamlit st.download_button.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A5276"),
        alignment=1,  # Center
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#566573"),
        alignment=1,
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2C3E50"),
        spaceAfter=6,
    )
    cell_style = ParagraphStyle(
        "CellStyle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["Italic"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#7F8C8D"),
        alignment=1,
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("<b>KORBA AIR QUALITY FORECAST BULLETIN</b>", title_style))
    story.append(Paragraph(f"Station: {station_name} | Issued: {issue_date} | Model: {model_version}", subtitle_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A5276"), spaceAfter=12))

    # 2. Extreme Spike Alert Banner (if triggered)
    if spike_alert and spike_alert.has_spike:
        alert_text = (
            f"<b>⚠️ {spike_alert.alert_title}</b>: {spike_alert.alert_message} "
            f"(Peak expected at Hour {spike_alert.peak_hour} - Category: {spike_alert.peak_category})."
        )
        alert_style = ParagraphStyle(
            "AlertStyle",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#922B21"),
        )
        alert_table = Table([[Paragraph(alert_text, alert_style)]], colWidths=[540])
        alert_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FADBD8")),
                ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#E74C3C")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ])
        )
        story.append(alert_table)
        story.append(Spacer(1, 14))

    # 3. Forecast Summary Table
    story.append(Paragraph("<b>1. Multi-Day Air Quality Forecast Summary</b>", section_style))
    table_data = [
        ["Forecast Period", "PM2.5 (ug/m3)", "PM10 (ug/m3)", "Dominant", "Predicted AQI", "Category"]
    ]

    for res in daily_results:
        table_data.append([
            res.day_name,
            f"{res.pm25_mean:.1f}",
            f"{res.pm10_mean:.1f}",
            res.dominant_pollutant,
            str(res.overall_aqi),
            res.category,
        ])

    summary_table = Table(table_data, colWidths=[90, 90, 90, 80, 90, 100])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980B9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9F9")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # 4. Demographic Health Advisories
    story.append(Paragraph("<b>2. Health Advisories & Precautionary Measures</b>", section_style))
    adv_data = [
        ["Day", "Category", "General Public Guidance", "Sensitive Demographics Guidance"]
    ]

    for adv in advisories:
        adv_data.append([
            adv.day_name,
            adv.category,
            Paragraph(adv.general_public_advice, cell_style),
            Paragraph(adv.sensitive_groups_advice, cell_style),
        ])

    adv_table = Table(adv_data, colWidths=[65, 75, 200, 200])
    adv_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(adv_table)
    story.append(Spacer(1, 20))

    # 5. Mandatory Medical Disclaimer & Data Attribution
    disclaimer_p = (
        "<b>Disclaimer:</b> This bulletin is generated for public informational and environmental preparedness purposes. "
        "It does not constitute formal medical advice. Individuals suffering from chronic respiratory, cardiovascular, or "
        "other medical vulnerabilities should strictly adhere to their personal physician's directives.<br/>"
        "Data Source: Continuous Ambient Air Quality Monitoring (CECB / CPCB)."
    )
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#BDC3C7"), spaceAfter=8))
    story.append(Paragraph(disclaimer_p, disclaimer_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
