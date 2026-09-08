from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUT = Path(__file__).resolve().parents[1] / "output" / "pdf" / "AethyxLM_Three_Person_Pitch_Script.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#10243E")
BLUE = colors.HexColor("#1E5AA8")
TEAL = colors.HexColor("#0B7A75")
GOLD = colors.HexColor("#D99A18")
LIGHT = colors.HexColor("#F2F6FA")
MID = colors.HexColor("#D9E4EF")
INK = colors.HexColor("#1D2A36")
MUTED = colors.HexColor("#536577")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=30, leading=35, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="CoverSub", parent=styles["Normal"], fontName="Helvetica",
    fontSize=13, leading=18, textColor=MUTED, alignment=TA_CENTER, spaceAfter=16,
))
styles.add(ParagraphStyle(
    name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=19, leading=23, textColor=NAVY, spaceBefore=8, spaceAfter=9,
))
styles.add(ParagraphStyle(
    name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=13, leading=17, textColor=BLUE, spaceBefore=8, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=10.3, leading=15, textColor=INK, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="Small", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.6, leading=12, textColor=MUTED, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Note", parent=styles["BodyText"], fontName="Helvetica-Oblique",
    fontSize=9, leading=13, textColor=MUTED, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Speaker", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=14, leading=18, textColor=TEAL, spaceBefore=10, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="Quote", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=11, leading=16, textColor=NAVY, leftIndent=10, rightIndent=10,
    borderColor=MID, borderWidth=0.6, borderPadding=8, backColor=LIGHT,
    spaceBefore=5, spaceAfter=9,
))


def P(text, style="BodyCustom"):
    return Paragraph(text.replace("\n", "<br/>"), styles[style])


def bullet(items):
    return [P(f"&bull; {item}", "BodyCustom") for item in items]


def role_box(role, focus, time):
    t = Table([[P(f"<b>{role}</b>", "BodyCustom"), P(focus, "BodyCustom"), P(time, "Small")]], colWidths=[39*mm, 119*mm, 22*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, MID),
        ("LINEBEFORE", (1, 0), (1, 0), 2, TEAL),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def note(text):
    return Table([[P(f"<b>Speaker note:</b> {text}", "Note")]], colWidths=[180*mm], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E9C66B")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(MID)
    canvas.line(16*mm, 13*mm, 194*mm, 13*mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(16*mm, 8*mm, "AethyxLM | Three-person principal pitch")
    canvas.drawRightString(194*mm, 8*mm, f"{doc.page}")
    canvas.restoreState()


story = []

# Cover
story += [Spacer(1, 30*mm), P("AETHYXLM", "CoverTitle"), P("Three-person pitch script", "CoverSub")]
story += [P("For presentation to a school principal", "CoverSub"), Spacer(1, 8*mm)]
cover = Table([[P("BUILD AI THAT CAN DO MORE WITH LESS", "H2Custom")]], colWidths=[150*mm])
cover.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.8, MID),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 12),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
]))
story += [cover, Spacer(1, 18*mm), P("Roles: CEO/CTO (main presenter), CIO (information and evaluation), CMO (users and adoption)", "Small"), P("Recommended duration: 6-8 minutes", "Small"), PageBreak()]

# Presentation guide
story += [P("How to present", "H1Custom"), P("The CEO/CTO carries the technical vision and the funding request. The CIO and CMO add credibility by explaining their areas of responsibility without taking over the core technology explanation.", "BodyCustom")]
story += [role_box("CEO / CTO", "Lead the problem, context-compression idea, model, evidence, roadmap, and funding request.", "4-5 min"), Spacer(1, 5), role_box("CIO", "Explain privacy-conscious information handling, evaluation discipline, and responsible execution.", "1 min"), Spacer(1, 5), role_box("CMO", "Explain target users, practical value, positioning, and adoption.", "1 min"), Spacer(1, 12)]
story += [P("Delivery rules", "H2Custom")]
story += bullet([
    "Speak conversationally; do not read every sentence with the same rhythm.",
    "Pause after the 63% result and after the funding request.",
    "Keep the distinction clear: the 63% figure is an early test result, not a guarantee.",
    "Present Rs 1,00,000 as Phase 1 funding, not the complete cost of the 50-billion-token training run.",
    "When challenged, return to milestones, measurements, and evidence.",
])
story += [P("Suggested handoff order", "H2Custom"), P("CEO/CTO opens -> CIO explains information and evaluation -> CMO explains users and adoption -> CEO/CTO closes with the roadmap and ask.", "BodyCustom"), PageBreak()]

# Script
story += [P("Presentation script", "H1Custom")]
story += [P("CEO / CTO - Opening and main technology", "Speaker"), note("Stand upright, make eye contact, and explain the context problem slowly. The audience may not know how AI applications provide conversation history, so use the plain-language explanation before introducing AethyxLM.")]
story += [P("Good morning, Sir/Ma’am.", "BodyCustom")]
story += [P("We are here to present AethyxLM, a student-led Indian AI project focused on making language models more efficient, affordable, and privacy-conscious.", "BodyCustom")]
story += [P("Let me begin with the problem.", "BodyCustom")]
story += [P("When we continue an AI conversation, the application usually sends relevant earlier messages, instructions, documents, and tool information to the model again so it has enough context to answer. The model does not automatically remember every conversation by itself.", "BodyCustom")]
story += [P("This repeated information increases usage cost, slows responses, and can expose more private data than necessary.", "BodyCustom")]
story += [P("That is the problem AethyxLM is trying to solve.", "BodyCustom")]
story += [P("We are not simply building another chatbot. We are building two connected systems: our own language model, and a context-preparation layer that makes information more efficient before it reaches the model.", "BodyCustom")]
story += [P("The context system works locally. It organises the user’s information, removes unnecessary repetition, preserves important facts, and sends only what is useful for the current question.", "BodyCustom")]
story += [P("In simple terms, the user’s device prepares the information, the model receives less unnecessary data, and the system can potentially respond faster and more cheaply.", "BodyCustom")]
story += [P("This is a research question: Can we represent the same useful information in a denser and more efficient form?", "Quote")]
story += [P("We have already conducted an initial 600-case test. In that test, the shortened context used 63% less model input and transferred 64% less data while retaining the important sources and key facts measured in that experiment.", "BodyCustom")]
story += [note("Slow down here. Say “early test” clearly. Do not present 63% as a guaranteed commercial saving.")]
story += [P("These are early results, not final commercial guarantees. We still need to test answer quality, latency, reliability, and actual cost more rigorously. But this result gives us a measurable reason to continue.", "BodyCustom")]
story += [P("Our current model has approximately 137.6 million parameters. It is being trained on English, Indian languages, coding data, mathematics, and general knowledge.", "BodyCustom")]
story += [P("Our next target is a 1-billion-parameter model trained on approximately 50 billion tokens.", "BodyCustom")]
story += [P("The purpose is not to make the model larger just for the sake of a larger number. The purpose is to test whether a larger model can provide better quality while working with our more efficient context system.", "BodyCustom")]
story += [P("Our first potential users are coding assistants working with software projects, multilingual assistants handling long conversations, document assistants processing policies and research, and business assistants managing long-running tasks.", "BodyCustom")]
story += [P("These products repeatedly handle large amounts of information, so reducing unnecessary context could make them cheaper and easier to scale.", "BodyCustom")]
story += [P("I’ll now hand over to our CIO, who will explain the information, privacy, and evaluation focus.", "BodyCustom")]

story += [P("CIO - Information, privacy, and evaluation", "Speaker"), note("Keep this section practical. Explain what you are responsible for, not the model architecture. Sound organised and careful.")]
story += [P("Thank you.", "BodyCustom")]
story += [P("My focus is making sure the project handles information responsibly and that our claims are properly measured.", "BodyCustom")]
story += [P("AethyxLM may eventually work with conversations, documents, application data, and long-running user histories. Because of that, we are focusing on local data preparation, privacy-conscious processing, and clear information-handling practices.", "BodyCustom")]
story += [P("I am also focusing on evaluation. We do not want to measure success only by how little data we send. We need to confirm that the useful information is still preserved.", "BodyCustom")]
story += bullet(["answer accuracy", "exact fact retention", "context retrieval", "latency, bandwidth, memory usage, and estimated cost"])
story += [P("My role is to make sure the project remains organised, measurable, and responsible as the model becomes larger.", "BodyCustom")]
story += [P("I’ll now hand over to our CMO, who will explain the users and practical opportunity.", "BodyCustom"), PageBreak()]

story += [P("CMO - Users, positioning, and adoption", "Speaker"), note("Focus on why the project matters to real users. Avoid technical detail. Make the audience understand the business and practical value.")]
story += [P("Thank you.", "BodyCustom")]
story += [P("My focus is understanding who can benefit from AethyxLM and how we communicate its value clearly.", "BodyCustom")]
story += [P("Our initial target users are applications that repeatedly handle large amounts of information: coding assistants, multilingual assistants, document tools, and business AI systems.", "BodyCustom")]
story += [P("These systems can become expensive and slow because they repeatedly process large histories and files. If AethyxLM can reduce unnecessary context while maintaining answer quality, it could help these products become more efficient and easier to scale.", "BodyCustom")]
story += [P("I am also focusing on positioning AethyxLM as more than just another language model. Our difference is the complete system: the model and the way information is prepared before reaching it.", "BodyCustom")]
story += [P("The project is being developed in India, but the problem we are addressing is global. Our goal is to turn the research into a clear working demonstration that people can understand, test, and eventually use.", "BodyCustom")]
story += [P("I’ll hand the presentation back to our CEO and CTO to explain the funding request and roadmap.", "BodyCustom"), PageBreak()]

story += [P("CEO / CTO - Funding, roadmap, and close", "Speaker"), note("Return to the centre of the presentation. Be specific about the money, milestones, and what the school is supporting. End calmly and confidently.")]
story += [P("Thank you.", "BodyCustom")]
story += [P("Our request is Rs 1,00,000 for the first development phase.", "Quote")]
story += [P("The proposed allocation is:", "BodyCustom")]
story += bullet(["45% for cloud computing and training", "25% for data preparation", "20% for model testing", "10% for basic operations"])
story += [P("This funding is not being presented as the complete cost of the entire 50-billion-token training run. It will help us build the next model version, run serious experiments, produce initial checkpoints, and generate reliable results for the next stage.", "BodyCustom")]
story += [P("Our immediate milestones are clear: finish evaluating the current model, begin the 1B-parameter training experiments, benchmark context compression, and demonstrate quality, speed, data usage, and cost in a live comparison.", "BodyCustom")]
story += [P("We are not asking for support based only on ambition. We are asking for support for a controlled project with specific milestones and measurable outcomes.", "BodyCustom")]
story += [P("If the results are weaker than expected, we will have evidence showing us what needs to improve. If the results are strong, we will have demonstrated a genuinely differentiated Indian AI platform designed to do more with less.", "BodyCustom")]
story += [P("AethyxLM is working on two connected challenges: building a capable language model, and finding a more efficient way for that model to consume information.", "BodyCustom")]
story += [P("The goal is not simply to build a bigger model. The goal is to build a smarter and more efficient AI system.", "BodyCustom")]
story += [P("With your support, we can take AethyxLM from a promising student-built project to a properly tested working demonstration.", "BodyCustom")]
story += [P("We are not asking you to fund a promise. We are asking you to help us run the experiments that prove whether this promise is real.", "Quote")]
story += [P("Thank you.", "BodyCustom"), PageBreak()]

# Q&A guide
story += [P("Likely questions and answers", "H1Custom")]
qa = [
    ("Why should I believe this will work?", "We are not presenting the hypothesis as a finished product. We already have an initial 600-case result, and the next funding is specifically for controlled testing of quality, speed, data usage, and cost."),
    ("Why does this need Rs 1 lakh?", "The largest cost is computing. Training a larger model requires cloud GPUs, while data preparation and evaluation are also essential. The Rs 1 lakh is for the first development phase, with clear milestones and measurable outputs."),
    ("Why not simply use an existing model?", "Existing models are useful, but they do not give us full control over the architecture, tokenizer, training data, or context-processing system. Building our own platform lets us research the complete system together."),
    ("What does the school gain?", "The school would support an ambitious student-led technology project with measurable outcomes. It would also demonstrate that the school supports original research, entrepreneurship, and applied AI development."),
    ("What if the project fails?", "Then we will still produce useful experimental evidence. The project is structured around checkpoints and benchmarks, so failure would identify what does not work instead of leaving the idea untested."),
    ("Are the 63% savings guaranteed?", "No. That figure comes from an early controlled test. We present it as evidence that the idea is worth investigating, not as a guaranteed commercial result."),
]
for q, a in qa:
    story += [P(q, "H2Custom"), P(a, "BodyCustom")]

story += [Spacer(1, 8), P("Final reminder", "H2Custom"), P("The CEO/CTO should own the technical story and the ask. The CIO and CMO should add confidence by showing that information handling, evaluation, users, and adoption are being considered seriously.", "BodyCustom")]

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
    topMargin=15*mm, bottomMargin=18*mm, title="AethyxLM Three-Person Pitch Script",
    author="Aethyx Labs",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
