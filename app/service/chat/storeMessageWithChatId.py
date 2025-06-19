from app.db.sql import AsyncSessionLocal
import sqlalchemy as sa
from datetime import datetime

async def store_message_and_reply(chat_id: str, message: str, reply: str):
    async with AsyncSessionLocal() as session:
        try:
            async with session.begin():  # Ensures transaction context
                # 1. Insert customer message
                await session.execute(
                    sa.text("""
                        INSERT INTO message (chatid, content, role, created_at)
                        VALUES (:chatid, :content, 'customer', :created_at)
                    """),
                    {
                        "chatid": chat_id,
                        "content": message,
                        "created_at": datetime.utcnow()
                    }
                )

                # 2. Insert agent reply
                await session.execute(
                    sa.text("""
                        INSERT INTO message (chatid, content, role, created_at)
                        VALUES (:chatid, :content, 'agent', :created_at)
                    """),
                    {
                        "chatid": chat_id,
                        "content": reply,
                        "created_at": datetime.utcnow()
                    }
                )

                # 3. Update last activity
                await session.execute(
                    sa.text("""
                        UPDATE chat
                        SET lastactivity = :updated_at
                        WHERE chatid = :chatid
                    """),
                    {
                        "chatid": chat_id,
                        "updated_at": datetime.utcnow()
                    }
                )

            return {"reply": reply}
        except Exception as e:
            print("Erreur lors de l'enregistrement du message:", e)
            return {"error": "Erreur serveur"}, 500