from datetime import datetime
from io import BytesIO
try:
    from fpdf import FPDF
except ImportError:
    # Fallback if fpdf2 is not installed yet
    FPDF = None


class CompliancePDF(FPDF):
    def header(self):
        # Logo placeholder or Title
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(15, 23, 42) # Dark Navy
        self.cell(0, 10, 'Fairness Audit & Compliance Report | Automated Decision Systems', border=False, ln=1, align='L')
        self.set_draw_color(22, 163, 74) # Accent Green
        self.line(10, 20, 200, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(100, 116, 139) # Text Muted
        self.cell(0, 10, f'Page {self.page_no()} | CONFIDENTIAL -- INTERNAL USE ONLY', align='C')


def generate_report(
    metrics_before,
    bias_before,
    metrics_after=None,
    bias_after=None,
    sensitive_col=None,
    model_type=None,
    mitigation_method=None,
):
    """
    Generate a structured Markdown compliance report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = []
    report.append("# Automated Fairness Audit & Compliance Report")
    report.append(f"\nGenerated: {timestamp}")
    report.append(f"\nClassification: CONFIDENTIAL -- INTERNAL USE ONLY")
    report.append("\n---\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(
        "This report presents the findings of an automated fairness audit conducted "
        "on a loan approval classification model. The audit evaluates model performance, "
        "detects demographic bias across protected attributes, applies mitigation "
        "techniques where necessary, and provides a compliance assessment.\n"
    )

    # Model Configuration
    report.append("## 1. Model Configuration\n")
    if model_type:
        report.append(f"- **Algorithm**: {model_type}")
    if sensitive_col:
        report.append(f"- **Protected Attribute**: {sensitive_col}")
    report.append(f"- **Audit Date**: {timestamp}")
    report.append("")

    # Initial Performance
    report.append("## 2. Baseline Model Performance\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    for k, v in metrics_before.items():
        report.append(f"| {k} | {v:.4f} |")
    report.append("")

    # Initial Bias Metrics
    report.append("## 3. Bias Detection Results\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    for k, v in bias_before.items():
        if isinstance(v, (int, float)):
            report.append(f"| {k} | {v:.4f} |")
        else:
            report.append(f"| {k} | {v} |")
    report.append("")

    di = bias_before.get("Disparate Impact", 1.0)
    if di < 0.8:
        report.append(
            "**Assessment**: The model exhibits significant bias. "
            "Disparate Impact is below the 0.80 threshold established by the "
            "EEOC four-fifths rule. Mitigation is strongly recommended.\n"
        )
    elif di < 0.9:
        report.append(
            "**Assessment**: The model shows moderate bias indicators. "
            "While above the strict 0.80 threshold, monitoring is advised.\n"
        )
    else:
        report.append(
            "**Assessment**: The model meets fairness criteria. "
            "Disparate Impact is within acceptable bounds.\n"
        )

    # Post-Mitigation
    if metrics_after and bias_after:
        report.append("## 4. Post-Mitigation Performance\n")
        if mitigation_method:
            report.append(f"**Mitigation Method**: {mitigation_method}\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        for k, v in metrics_after.items():
            report.append(f"| {k} | {v:.4f} |")
        report.append("")

        report.append("## 5. Post-Mitigation Fairness Metrics\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        for k, v in bias_after.items():
            if isinstance(v, (int, float)):
                report.append(f"| {k} | {v:.4f} |")
            else:
                report.append(f"| {k} | {v} |")
        report.append("")

        di_after = bias_after.get("Disparate Impact", 1.0)
        if di_after >= 0.8:
            report.append(
                "**Result**: Mitigation successfully brought the model into "
                "compliance with the four-fifths rule.\n"
            )
        else:
            report.append(
                "**Result**: Despite mitigation, the model remains below the "
                "compliance threshold. Additional intervention is recommended.\n"
            )

        # Performance impact
        acc_before = metrics_before.get("Accuracy", 0)
        acc_after = metrics_after.get("Accuracy", 0)
        acc_delta = acc_after - acc_before
        report.append("## 6. Performance Impact Analysis\n")
        report.append(f"- Accuracy change: {acc_delta:+.4f}")
        report.append(
            f"- Disparate Impact improvement: "
            f"{di_after - di:+.4f}"
        )
        report.append("")

    # Regulatory References
    report.append("## Regulatory References\n")
    report.append("- Equal Credit Opportunity Act (ECOA)")
    report.append("- Fair Housing Act (FHA)")
    report.append("- EEOC Uniform Guidelines (Four-Fifths Rule)")
    report.append("- EU AI Act -- High-Risk Classification Systems")
    report.append("")

    report.append("---")
    report.append(
        "\n*This report was generated by the Fairness Audit & Bias Mitigation Pipeline v2.0*"
    )

    return "\n".join(report)


def generate_pdf_report(
    metrics_before,
    bias_before,
    metrics_after=None,
    bias_after=None,
    sensitive_col=None,
    model_type=None,
    mitigation_method=None,
):
    """
    Generate a formatted PDF compliance report.
    """
    if not FPDF:
        return None

    pdf = CompliancePDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Title Section
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(17, 24, 39) # Primary Text
    pdf.cell(0, 10, 'Executive Summary', ln=1)
    
    pdf.set_font('helvetica', '', 10)
    summary_text = (
        "This document details the fairness and performance audit of the loan approval "
        "automated decision system. The audit covers baseline model analysis and, if applicable, "
        "the outcomes of algorithmic bias mitigation strategies to ensure regulatory compliance."
    )
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(5)

    # 2. Configuration Info
    pdf.set_fill_color(248, 250, 252) # BG secondary
    pdf.rect(10, pdf.get_y(), 190, 22, 'F')
    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(40, 6, 'Algorithm:')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 6, f'{model_type or "N/A"}', ln=1)
    
    pdf.set_x(12)
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(40, 6, 'Protected Attribute:')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 6, f'{sensitive_col or "N/A"}', ln=1)
    
    pdf.set_x(12)
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(40, 6, 'Audit Timestamp:')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 6, f'{timestamp}', ln=1)
    pdf.ln(8)

    # 3. Before and After Comparison Table (The Requested Feature)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Model Performance & Fairness Comparison', ln=1)
    
    # Table Header
    pdf.set_fill_color(15, 23, 42) # Dark Navy
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('helvetica', 'B', 9)
    col_width = [60, 45, 45, 40]
    pdf.cell(col_width[0], 8, ' Metric', border=1, fill=True)
    pdf.cell(col_width[1], 8, ' Baseline', border=1, fill=True, align='C')
    pdf.cell(col_width[2], 8, ' Mitigated', border=1, fill=True, align='C')
    pdf.cell(col_width[3], 8, ' Delta', border=1, fill=True, align='C')
    pdf.ln()

    # Table Body
    pdf.set_text_color(17, 24, 39)
    pdf.set_font('helvetica', '', 9)
    
    # Combine all metrics for comparison
    all_metric_names = list(metrics_before.keys()) + list(bias_before.keys())
    
    for m in all_metric_names:
        val_before = metrics_before.get(m) if m in metrics_before else bias_before.get(m)
        val_after = None
        if metrics_after and m in metrics_after: val_after = metrics_after.get(m)
        elif bias_after and m in bias_after: val_after = bias_after.get(m)
        
        # Formatting
        str_before = f"{val_before:.4f}" if isinstance(val_before, (float, int)) else str(val_before)
        str_after = f"{val_after:.4f}" if isinstance(val_after, (float, int)) else (str(val_after) if val_after is not None else "--")
        
        delta_val = "--"
        if val_after is not None and isinstance(val_before, (float, int)) and isinstance(val_after, (float, int)):
            delta = val_after - val_before
            delta_val = f"{delta:+.4f}"
            if m == "Disparate Impact" and delta > 0: pdf.set_text_color(22, 163, 74) # Improvement Blue/Green
            elif m == "Accuracy" and delta < 0: pdf.set_text_color(220, 38, 38) # Decrease Red
        else:
            pdf.set_text_color(17, 24, 39)

        pdf.cell(col_width[0], 7, f' {m}', border=1)
        pdf.cell(col_width[1], 7, str_before, border=1, align='C')
        pdf.cell(col_width[2], 7, str_after, border=1, align='C')
        pdf.cell(col_width[3], 7, delta_val, border=1, align='C')
        pdf.ln()
        pdf.set_text_color(17, 24, 39) # Reset

    pdf.ln(10)

    # 4. Compliance Assessment
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Final Compliance Assessment', ln=1)
    pdf.set_font('helvetica', '', 10)
    
    di_final = bias_after.get("Disparate Impact") if bias_after else bias_before.get("Disparate Impact", 1.0)
    assessment = ""
    if di_final < 0.8:
        assessment = "FAILED: Model exhibits significant disparate impact. Does not meet EEOC Four-Fifths Rule."
    elif di_final < 0.9:
        assessment = "WARNING: Model meets basic compliance but shows moderate demographic variance."
    else:
        assessment = "PASSED: Model is considered fair and compliant with regulatory standards."
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.multi_cell(0, 8, assessment, border=1)
    pdf.ln(10)

    # 5. Regulatory Disclaimers
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 8, 'Regulatory Frameworks Referenced:', ln=1)
    pdf.set_font('helvetica', '', 8)
    frameworks = [
        "Equal Credit Opportunity Act (ECOA) - 15 U.S.C. 1691",
        "Fair Housing Act (FHA) - 42 U.S.C. 3601",
        "EEOC Uniform Guidelines on Employee Selection Procedures (Four-Fifths Rule)",
        "EU AI Act Requirements for High-Risk Algorithmic Decision Systems",
        "OCC 2011-12 / Federal Reserve SR 11-7 Model Risk Management Guidance"
    ]
    for fw in frameworks:
        pdf.cell(10)
        pdf.cell(0, 5, f"- {fw}", ln=1)

    # Return as bytes
    return bytes(pdf.output())
