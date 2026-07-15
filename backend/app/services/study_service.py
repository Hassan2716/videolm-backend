"""Study asset generation — flashcards, quiz, mind map, notes."""
import uuid
from loguru import logger
from app.core.database import SessionLocal, StudyAsset, Transcript


class StudyService:
    @staticmethod
    def generate(project_id: str, asset_type: str):
        db = SessionLocal()
        try:
            t = db.query(Transcript).filter(Transcript.project_id == project_id).first()
            if not t:
                return
            fns = {
                "flashcard": StudyService._flashcards,
                "quiz": StudyService._quiz,
                "mindmap": StudyService._mindmap,
                "notes": StudyService._notes,
            }
            fn = fns.get(asset_type)
            content = fn(t.full_text) if fn else {}
            db.add(StudyAsset(id=str(uuid.uuid4()), project_id=project_id, asset_type=asset_type, content=content))
            db.commit()
            logger.info(f"Study asset: {asset_type} for {project_id}")
        finally:
            db.close()

    @staticmethod
    def _flashcards(text):
        sentences = [s.strip() for s in text.replace(".", ".\n").split("\n") if len(s.strip()) > 30]
        cards = []
        for sent in sentences[:15]:
            words = sent.split()
            if len(words) > 5:
                key_idx = len(words) // 2
                answer = words[key_idx]
                question = " ".join(words[:key_idx] + ["___"] + words[key_idx+1:])
                cards.append({"question": question, "answer": answer, "hint": sent})
        return {"flashcards": cards}

    @staticmethod
    def _quiz(text):
        sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 40][:10]
        questions = []
        for sent in sentences[:5]:
            words = sent.split()
            if len(words) > 8:
                questions.append({
                    "question": f"What is being described: '{sent[:60]}…'?",
                    "options": [f"A) {sent[:30]}", "B) Related concept", "C) Different topic", "D) None of above"],
                    "answer": "A",
                    "explanation": sent,
                })
        return {"questions": questions}

    @staticmethod
    def _mindmap(text):
        from app.pipeline.nlp.chunker import KeyphraseExtractor
        keyphrases = KeyphraseExtractor().extract(text, top_n=20)
        sentences = text.split(". ")
        topic = keyphrases[0] if keyphrases else "Main Topic"
        branches = []
        for phrase in keyphrases[1:6]:
            related = [s[:60] for s in sentences if phrase.lower() in s.lower()][:2]
            branches.append({"name": phrase, "children": related})
        return {"root": topic, "branches": branches}

    @staticmethod
    def _notes(text):
        from app.pipeline.nlp.chunker import KeyphraseExtractor
        keyphrases = KeyphraseExtractor().extract(text, top_n=10)
        sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 30]
        return {"key_concepts": keyphrases, "key_points": sentences[:8], "word_count": len(text.split())}
