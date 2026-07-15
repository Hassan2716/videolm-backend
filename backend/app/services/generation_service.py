"""Generation Service v2 — on-demand lazy generation. All features fixed."""
import os, uuid, re, math
from typing import Dict, Optional, List
from datetime import datetime
from loguru import logger
from app.core.database import SessionLocal, Project, Summary, StudyAsset, Transcript
from app.core.config import settings

_jobs: Dict = {}

def create_generation_job(project_id, job_type, params=None):
    jid = str(uuid.uuid4())
    _jobs[jid] = {"id":jid,"project_id":project_id,"type":job_type,"status":"pending",
                  "progress":0,"message":"Queued…","result":None,"error":None,
                  "params":params or {},"created_at":datetime.utcnow().isoformat()}
    return jid

def get_job_status(jid): return _jobs.get(jid)

def update_job(jid, **kw):
    if jid in _jobs: _jobs[jid].update(kw)

# ── Summary ───────────────────────────────────────────────────────────────────
async def generate_summary_async(job_id, project_id, summary_type, model_key):
    try:
        update_job(job_id, status="running", progress=10, message="Loading transcript…")
        db = SessionLocal()
        t = db.query(Transcript).filter(Transcript.project_id==project_id).first()
        db.close()
        if not t or not t.full_text:
            update_job(job_id, status="failed", error="Transcript not ready"); return
        update_job(job_id, progress=30, message="Running map-reduce summarization…")
        from app.pipeline.nlp.summarizer_v2 import HierarchicalSummarizer
        content = HierarchicalSummarizer(device=settings.device).summarize(
            text=t.full_text, summary_type=summary_type,
            model_key=model_key, segments=t.segments)
        if not content or len(content.strip()) < 20:
            content = "Summary generation produced insufficient content. Please try again."
        update_job(job_id, progress=85, message="Saving…")
        db = SessionLocal()
        ex = db.query(Summary).filter(Summary.project_id==project_id,
                                       Summary.summary_type==summary_type,
                                       Summary.model_used==model_key).first()
        if ex: ex.content=content; ex.word_count=len(content.split())
        else: db.add(Summary(id=str(uuid.uuid4()),project_id=project_id,
                             summary_type=summary_type,model_used=model_key,
                             content=content,word_count=len(content.split())))
        db.commit(); db.close()
        update_job(job_id, status="complete", progress=100, message="Done!",
                   result={"content":content,"word_count":len(content.split())})
    except Exception as e:
        logger.exception(f"Summary job failed: {e}")
        update_job(job_id, status="failed", error=str(e))

# ── Quiz ──────────────────────────────────────────────────────────────────────
async def generate_quiz_async(job_id, project_id, params):
    try:
        update_job(job_id, status="running", progress=10, message="Loading transcript…")
        db = SessionLocal()
        t = db.query(Transcript).filter(Transcript.project_id==project_id).first()
        db.close()
        if not t: update_job(job_id, status="failed", error="Transcript not ready"); return
        update_job(job_id, progress=30, message="Extracting key concepts…")
        from app.services.quiz_generator_v2 import QuizGenerator
        quiz = QuizGenerator().generate(
            text=t.full_text, segments=t.segments,
            num_questions=params.get("num_questions",10),
            difficulty=params.get("difficulty","medium"),
            question_types=params.get("question_types",["mcq","true_false","fill_blank"]))
        update_job(job_id, progress=80, message="Saving quiz…")
        db = SessionLocal()
        db.query(StudyAsset).filter(StudyAsset.project_id==project_id,
                                     StudyAsset.asset_type=="quiz").delete()
        db.add(StudyAsset(id=str(uuid.uuid4()),project_id=project_id,
                          asset_type="quiz",content=quiz))
        db.commit(); db.close()
        update_job(job_id, status="complete", progress=100,
                   message=f"{len(quiz.get('questions',[]))} questions ready!", result=quiz)
    except Exception as e:
        logger.exception(f"Quiz job failed: {e}")
        update_job(job_id, status="failed", error=str(e))

# ── Flashcards ────────────────────────────────────────────────────────────────
async def generate_flashcards_async(job_id, project_id, params):
    try:
        update_job(job_id, status="running", progress=15, message="Extracting key concepts…")
        db = SessionLocal()
        t = db.query(Transcript).filter(Transcript.project_id==project_id).first()
        db.close()
        if not t:
            update_job(job_id, status="failed", error="Transcript not ready"); return

        from app.services.quiz_generator_v2 import extract_concepts, _filter_transcript

        filtered = _filter_transcript(t.full_text)
        if len(filtered.split()) < 50:
            filtered = t.full_text

        update_job(job_id, progress=40, message="Building flashcards from concepts…")
        num_cards = params.get("num_cards", 20)
        concepts = extract_concepts(filtered, top_n=num_cards)

        import re
        sentences = re.split(r'(?<=[.!?])\s+', filtered)

        cards = []
        for concept in concepts:
            # Find ALL sentences containing this concept
            matching = [s.strip() for s in sentences
                       if concept.lower() in s.lower() and len(s.split()) >= 8]
            if not matching:
                continue

            # Use the LONGEST matching sentence as definition (most complete)
            definition = max(matching, key=lambda s: len(s))

            # Ensure sentence is complete — must end with . ! or ?
            if definition and definition[-1] not in '.!?':
                # Find next sentence boundary in original text
                idx = filtered.find(definition)
                if idx >= 0:
                    remaining = filtered[idx + len(definition):]
                    end = min(
                        (remaining.find(c) for c in '.!?' if c in remaining),
                        default=-1
                    )
                    if end >= 0 and end < 150:
                        definition = definition + remaining[:end+1]
                definition = definition.rstrip(', ') + '...'

            # Build context from surrounding sentences (complete sentences only)
            context_sentences = [s.strip() for s in matching[1:3]
                                 if s.strip() != definition and len(s.split()) >= 6
                                 and s.strip()[-1] in '.!?\n']

            back_parts = [definition]
            if context_sentences:
                back_parts.append("Also: " + context_sentences[0])

            back = "\n\n".join(back_parts)

            cards.append({
                "id": str(uuid.uuid4()),
                "front": f"What is {concept}?",
                "back": back,
                "concept": concept,
            })
            if len(cards) >= num_cards:
                break

        if not cards:
            update_job(job_id, status="failed",
                      error="Could not extract enough concepts. Ensure transcript is available.")
            return

        update_job(job_id, progress=80, message=f"Saving {len(cards)} flashcards…")
        db = SessionLocal()
        db.query(StudyAsset).filter(
            StudyAsset.project_id==project_id,
            StudyAsset.asset_type=="flashcard"
        ).delete()
        db.add(StudyAsset(
            id=str(uuid.uuid4()), project_id=project_id,
            asset_type="flashcard",
            content={"flashcards": cards, "total": len(cards)}
        ))
        db.commit(); db.close()
        update_job(job_id, status="complete", progress=100,
                   message=f"{len(cards)} flashcards ready!",
                   result={"flashcards": cards, "total": len(cards)})

    except Exception as e:
        logger.exception(f"Flashcard job failed: {e}")
        update_job(job_id, status="failed", error=str(e))

async def generate_mindmap_async(job_id, project_id, params):
    try:
        update_job(job_id, status="running", progress=20, message="Analysing content…")
        db = SessionLocal()
        t = db.query(Transcript).filter(Transcript.project_id==project_id).first()
        db.close()
        if not t: update_job(job_id, status="failed", error="No transcript available"); return
        from app.services.quiz_generator_v2 import extract_concepts, _filter_transcript
        filtered = _filter_transcript(t.full_text) or t.full_text
        concepts = extract_concepts(filtered, top_n=25)
        update_job(job_id, progress=50, message="Building mind map…")
        mindmap = _build_mindmap(t.full_text, concepts)
        db = SessionLocal()
        db.query(StudyAsset).filter(StudyAsset.project_id==project_id,
                                     StudyAsset.asset_type=="mindmap").delete()
        db.add(StudyAsset(id=str(uuid.uuid4()),project_id=project_id,
                          asset_type="mindmap",content=mindmap))
        db.commit(); db.close()
        update_job(job_id, status="complete", progress=100,
                   message="Mind map ready!", result=mindmap)
    except Exception as e:
        logger.exception(f"Mindmap job failed: {e}")
        update_job(job_id, status="failed", error=str(e))

def _build_mindmap(text, concepts):
    if not concepts:
        return {"nodes":[],"edges":[],"root_topic":"Topic","total_concepts":0}
    main = concepts[0]
    branches = concepts[1:7]
    leaves = concepts[7:25]
    nodes, edges = [], []
    nodes.append({"id":"root","type":"mindmapRoot","data":{"label":main,"level":0},
                  "position":{"x":500,"y":300},
                  "style":{"background":"#2563eb","color":"#fff","borderRadius":"14px",
                           "padding":"12px 22px","fontWeight":"700","fontSize":"15px",
                           "boxShadow":"0 4px 20px rgba(37,99,235,0.4)"}})
    n = len(branches)
    for i, branch in enumerate(branches):
        angle = (2*math.pi*i/max(n,1)) - math.pi/2
        bx = 500 + 230*math.cos(angle)
        by = 300 + 230*math.sin(angle)
        bid = f"branch_{i}"
        nodes.append({"id":bid,"type":"mindmapBranch","data":{"label":branch,"level":1},
                      "position":{"x":bx,"y":by},
                      "style":{"background":"#7c3aed","color":"#fff","borderRadius":"10px",
                               "padding":"8px 16px","fontWeight":"600","fontSize":"13px"}})
        edges.append({"id":f"e_root_{bid}","source":"root","target":bid,
                      "type":"smoothstep","style":{"stroke":"#94a3b8","strokeWidth":2}})
        for j, leaf in enumerate(leaves[i*3:(i+1)*3]):
            la = angle + 0.35*(j-1)
            lid = f"leaf_{i}_{j}"
            nodes.append({"id":lid,"type":"mindmapLeaf","data":{"label":leaf,"level":2},
                          "position":{"x":bx+130*math.cos(la),"y":by+130*math.sin(la)},
                          "style":{"background":"#0f172a","color":"#e2e8f0",
                                   "border":"1px solid #334155","borderRadius":"8px",
                                   "padding":"5px 12px","fontSize":"11px"}})
            edges.append({"id":f"e_{bid}_{lid}","source":bid,"target":lid,
                          "type":"smoothstep","style":{"stroke":"#475569","strokeWidth":1.5}})
    return {"nodes":nodes,"edges":edges,"root_topic":main,"total_concepts":len(concepts)}
