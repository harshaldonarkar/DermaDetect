"""
Generates a PDF diagnosis report for DermaDetect.
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, red, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable,
)
from PIL import Image as PILImage
import numpy as np

# Brand colours
GREEN_DARK  = HexColor("#175810")
GREEN_MID   = HexColor("#2f6329")
GREEN_LIGHT = HexColor("#5da64e")
BG_LIGHT    = HexColor("#f4fbf3")


def _pil_to_rl_image(pil_img, width_cm, height_cm):
    """Convert a PIL image to a ReportLab Image flowable."""
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def _badge_color(diagnosis):
    if "Non-Cancerous" in diagnosis:
        return HexColor("#1e7e34"), HexColor("#e8f8e8")
    if "lead" in diagnosis:
        return HexColor("#856404"), HexColor("#fff3cd")
    return HexColor("#c0392b"), HexColor("#fde8e8")


def generate_pdf(
    patient_name, patient_id, patient_age, patient_sex,
    patient_ethnicity, medical_history,
    diagnosis, confidence, low_confidence, info,
    original_pil, gradcam_pil,
):
    """
    Build and return a PDF report as bytes.

    Parameters
    ----------
    original_pil : PIL.Image  — the uploaded skin image (224×224)
    gradcam_pil  : PIL.Image  — the Grad-CAM overlay image (224×224)
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    style_h1 = ParagraphStyle(
        "h1", fontSize=26, textColor=GREEN_DARK,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=2,
    )
    style_h2 = ParagraphStyle(
        "h2", fontSize=13, textColor=GREEN_MID,
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4,
    )
    style_body = ParagraphStyle(
        "body", fontSize=10, textColor=black,
        fontName="Helvetica", leading=15, alignment=TA_JUSTIFY,
    )
    style_small = ParagraphStyle(
        "small", fontSize=8.5, textColor=HexColor("#555555"),
        fontName="Helvetica", alignment=TA_CENTER,
    )
    style_disclaimer = ParagraphStyle(
        "disclaimer", fontSize=9, textColor=red,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceBefore=10,
    )
    style_center = ParagraphStyle(
        "center", fontSize=10, fontName="Helvetica", alignment=TA_CENTER,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("DermaDetect", style_h1))
    story.append(Paragraph(
        "AI-Assisted Skin Lesion Diagnosis Report",
        ParagraphStyle("sub", fontSize=11, textColor=GREEN_LIGHT,
                       fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4),
    ))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}",
        style_small,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN_LIGHT, spaceAfter=14))

    # ── Patient Information ───────────────────────────────────────────────────
    story.append(Paragraph("Patient Information", style_h2))

    patient_data = [
        ["Name", patient_name or "—",       "Patient ID", patient_id or "—"],
        ["Age",  patient_age or "—",         "Sex",        (patient_sex or "—").capitalize()],
        ["Ethnicity", (patient_ethnicity or "—").replace("_", " ").capitalize(),
         "Medical History", medical_history or "None provided"],
    ]

    patient_table = Table(patient_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 4.5*cm])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), BG_LIGHT),
        ("BACKGROUND",  (2, 0), (2, -1), BG_LIGHT),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, HexColor("#f9f9f9")]),
        ("BOX",         (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, HexColor("#dddddd")),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # ── Diagnosis Result ─────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cccccc")))
    story.append(Paragraph("Diagnosis", style_h2))

    text_color, bg_color = _badge_color(diagnosis)
    diag_table = Table(
        [[Paragraph(f"<b>{diagnosis}</b>",
                    ParagraphStyle("d", fontSize=13, textColor=text_color,
                                   fontName="Helvetica-Bold", alignment=TA_CENTER))]],
        colWidths=[17.5 * cm],
    )
    diag_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg_color),
        ("BOX",           (0, 0), (-1, -1), 1, text_color),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 8))

    # Confidence bar (text representation)
    bar_filled = int(confidence / 5)
    bar_empty  = 20 - bar_filled
    bar_str    = "█" * bar_filled + "░" * bar_empty
    story.append(Paragraph(
        f"Model Confidence: <b>{confidence}%</b>  {bar_str}",
        ParagraphStyle("conf", fontSize=10, fontName="Helvetica",
                       alignment=TA_CENTER, textColor=HexColor("#333333")),
    ))

    if low_confidence:
        story.append(Spacer(1, 6))
        warn_table = Table(
            [[Paragraph(
                "⚠  Confidence below 60% — result may be unreliable. "
                "Dermatologist consultation strongly advised.",
                ParagraphStyle("w", fontSize=9, textColor=HexColor("#856404"),
                               fontName="Helvetica", alignment=TA_CENTER),
            )]],
            colWidths=[17.5 * cm],
        )
        warn_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#fff3cd")),
            ("BOX",        (0, 0), (-1, -1), 0.5, HexColor("#ffc107")),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(warn_table)

    story.append(Spacer(1, 14))

    # ── Images ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Skin Lesion Analysis", style_h2))

    orig_rl    = _pil_to_rl_image(original_pil, 7.5, 7.5)
    gradcam_rl = _pil_to_rl_image(gradcam_pil,  7.5, 7.5)

    img_label = ParagraphStyle(
        "img_label", fontSize=9, fontName="Helvetica-Bold",
        alignment=TA_CENTER, textColor=HexColor("#444444"),
    )
    img_note = ParagraphStyle(
        "img_note", fontSize=7.5, fontName="Helvetica",
        alignment=TA_CENTER, textColor=HexColor("#777777"),
    )

    img_table = Table(
        [
            [orig_rl, gradcam_rl],
            [Paragraph("Original Image", img_label),
             Paragraph("Grad-CAM Heatmap", img_label)],
            [Paragraph("Uploaded skin lesion", img_note),
             Paragraph("Red = region most influential to diagnosis", img_note)],
        ],
        colWidths=[8.75 * cm, 8.75 * cm],
    )
    img_table.setStyle(TableStyle([
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 14))

    # ── Medical Information ───────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cccccc")))
    story.append(Paragraph("About This Condition", style_h2))
    story.append(Paragraph(info, style_body))
    story.append(Spacer(1, 16))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=red, spaceAfter=8))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI model and is intended for "
        "informational purposes only. It does not constitute a medical diagnosis. "
        "Please consult a qualified dermatologist for professional medical advice and treatment.",
        style_disclaimer,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph("DermaDetect — AI Skin Cancer Detection System", style_small))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
