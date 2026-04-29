from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ieee_tox21_mlops_report.pdf"


def scaled_image(path, width):
    image = Image(str(path))
    ratio = image.imageHeight / float(image.imageWidth)
    image.drawWidth = width
    image.drawHeight = width * ratio
    image.hAlign = "CENTER"
    return image


def table(data, widths=None):
    tbl = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#777777")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def caption(text, styles):
    return Paragraph(text, styles["caption"])


def section(title, styles):
    return Paragraph(title, styles["section"])


def para(text, styles):
    return Paragraph(text, styles["body"])


def build():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="title_ieee",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="author",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="body",
            parent=styles["BodyText"],
            fontSize=9,
            leading=11,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="caption",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceBefore=3,
            spaceAfter=8,
        )
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Tox21 Molecular Toxicity Prediction Using an MLOps Deployment Pipeline",
    )

    story = []
    content_width = letter[0] - doc.leftMargin - doc.rightMargin

    story.append(
        Paragraph(
            "Tox21 Molecular Toxicity Prediction Using an MLOps Deployment Pipeline",
            styles["title_ieee"],
        )
    )
    story.append(
        Paragraph(
            "Author Name: TODO<br/>Department / Program: TODO<br/>Institution: TODO<br/>Email: TODO",
            styles["author"],
        )
    )
    story.append(section("Abstract", styles))
    story.append(
        para(
            "This report presents an end-to-end MLOps pipeline for molecular toxicity prediction using the Tox21 dataset. "
            "The project converts molecular SMILES strings into 1024-bit Morgan fingerprints with RDKit, trains and compares "
            "Logistic Regression, Random Forest, and Extra Trees classifiers, tracks experiments with MLflow, serves the selected "
            "model through FastAPI, and exposes operational metrics through Prometheus and Grafana.",
            styles,
        )
    )
    story.append(section("Keywords", styles))
    story.append(
        para(
            "MLOps, Tox21, molecular toxicity prediction, FastAPI, Docker, MLflow, Prometheus, Grafana, GitHub Actions.",
            styles,
        )
    )

    story.append(section("I. Introduction", styles))
    story.append(
        para(
            "This project focuses on a reproducible MLOps workflow for the Tox21 SR-ARE toxicity endpoint. "
            "The repository includes the dataset, training pipeline, selected model artifact, FastAPI inference service, "
            "Docker configuration, Prometheus metrics, Grafana dashboard provisioning, CI validation, and CD image publishing workflow.",
            styles,
        )
    )

    story.append(section("II. System Architecture", styles))
    story.append(scaled_image(ROOT / "High_Level_Design.png", content_width))
    story.append(
        caption(
            "Fig. 1. High-level architecture of the Tox21 MLOps pipeline for molecular toxicity prediction.",
            styles,
        )
    )
    story.append(
        para(
            "The architecture is organized around data and model development, Docker-based deployment and orchestration, "
            "and client observability. The API exposes prediction, health, version, and metrics endpoints.",
            styles,
        )
    )

    story.append(section("III. Dataset and Feature Engineering", styles))
    story.append(
        table(
            [
                ["Item", "Value"],
                ["Dataset", "Tox21"],
                ["Target", "SR-ARE"],
                ["Input", "SMILES string"],
                ["Feature type", "Morgan fingerprints"],
                ["Fingerprint size", "1024 bits"],
                ["Fingerprint radius", "2"],
            ],
            [2.1 * inch, 4.3 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        para(
            "The training script removes records with missing target labels and skips molecules that RDKit cannot parse. "
            "The same fingerprint representation is used during model training and API inference.",
            styles,
        )
    )

    story.append(section("IV. Model Training and Selection", styles))
    story.append(
        table(
            [
                ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                ["Logistic Regression", "0.7494", "0.3443", "0.6117", "0.4406", "0.7443"],
                ["Random Forest", "0.7923", "0.3846", "0.4787", "0.4265", "0.7592"],
                ["Extra Trees", "0.7700", "0.3571", "0.5319", "0.4274", "0.7648"],
            ],
            [1.65 * inch, 0.95 * inch, 0.95 * inch, 0.85 * inch, 0.75 * inch, 0.95 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        para(
            "Extra Trees is selected as the best model using ROC-AUC as the selection metric. The selected artifact is stored at "
            "models/tox21_best_model.joblib, with metadata stored in models/model_metadata.json.",
            styles,
        )
    )
    story.append(KeepTogether([scaled_image(ROOT / "Images_ForGPT" / "MLFLOW.PNG", content_width * 0.78), caption("Fig. 2. MLflow experiment tracking screenshot captured from the project artifacts.", styles)]))
    story.append(KeepTogether([scaled_image(ROOT / "Images_ForGPT" / "Grafana.PNG", content_width * 0.78), caption("Fig. 3. Grafana dashboard screenshot captured from the project artifacts.", styles)]))

    story.append(section("V. API Serving Layer", styles))
    story.append(
        table(
            [
                ["Endpoint", "Purpose"],
                ["GET /", "Service root message"],
                ["GET /health", "Model and service health metadata"],
                ["GET /version", "App, model, and runtime version metadata"],
                ["POST /predict", "Toxicity prediction for a SMILES input"],
                ["GET /metrics", "Prometheus-compatible service metrics"],
            ],
            [1.8 * inch, 4.6 * inch],
        )
    )

    story.append(section("VI. Observability", styles))
    story.append(
        para(
            "The API exposes Prometheus metrics for prediction attempts, outcomes, errors, response status codes, latency, and model load status. "
            "Grafana provisioning is included in the repository and groups panels around service health, traffic quality, and latency.",
            styles,
        )
    )

    story.append(section("VII. CI/CD and Reproducibility", styles))
    story.append(
        para(
            "The CI workflow installs pinned dependencies, verifies required files, compiles Python files, runs tests, checks model artifact compatibility, "
            "and builds the Docker image on pushes. The CD workflow publishes the API image to GitHub Container Registry and supports semantic version tags.",
            styles,
        )
    )

    story.append(section("VIII. Limitations", styles))
    story.append(
        para(
            "The project is intended as an MLOps demonstration, not a validated toxicology decision system. The model is specific to the SR-ARE endpoint "
            "and uses fixed fingerprint features. Real-world safety-critical use would require stronger validation, calibration analysis, data governance, "
            "monitoring, and domain expert review.",
            styles,
        )
    )

    story.append(section("IX. Conclusion", styles))
    story.append(
        para(
            "The project demonstrates a complete MLOps workflow for molecular toxicity prediction: preprocessing, feature generation, model comparison, "
            "experiment tracking, artifact management, API serving, containerization, observability, CI validation, and image publishing.",
            styles,
        )
    )

    story.append(section("Missing Information To Complete Final Submission", styles))
    story.append(
        para(
            "TODO: author name, institution or department, course or module name, instructor or supervisor name, submission date, and required citation style.",
            styles,
        )
    )

    doc.build(story)


if __name__ == "__main__":
    build()
