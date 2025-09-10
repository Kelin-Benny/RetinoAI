from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import base64
from datetime import datetime

def generate_pdf_report(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("RetinoAI Pro - Diagnostic Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Date and patient info
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(f"Report Date: {current_date}", styles['Normal']))
    
    if data.get('patient_data'):
        patient = data['patient_data']
        if patient.get('age'):
            story.append(Paragraph(f"Patient Age: {patient['age']}", styles['Normal']))
        if patient.get('diabetic'):
            story.append(Paragraph(f"Diabetic: {patient['diabetic'].capitalize()}", styles['Normal']))
        if patient.get('family_history'):
            story.append(Paragraph(f"Family History: {patient['family_history'].capitalize()}", styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Results
    story.append(Paragraph("Diagnostic Results", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    results_data = [
        ["Image Modality", f"{data['modality']} ({data['modality_confidence']}% confidence)"],
        ["Diagnosis", f"{data['diagnosis']} ({data['confidence']}% confidence)"]
    ]
    
    results_table = Table(results_data, colWidths=[2*inch, 4*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONT', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Probability distribution
    story.append(Paragraph("Probability Distribution", styles['Heading3']))
    story.append(Spacer(1, 0.2*inch))
    
    prob_data = [["Condition", "Probability (%)"]]
    for condition, probability in data['probabilities'].items():
        prob_data.append([condition, f"{probability}%"])
    
    prob_table = Table(prob_data, colWidths=[3*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONT', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    
    story.append(prob_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Risk assessment
    story.append(Spacer(1, 0.3*inch))
    normal_conditions = ['NORMAL', 'Normal']
    if data['diagnosis'] not in normal_conditions:
        story.append(Paragraph("Priority Recommendation", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(
            "URGENT: This case requires immediate specialist attention. " +
            "Please consult with a retinal specialist for further evaluation and treatment planning.",
            styles['Normal']
        ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer