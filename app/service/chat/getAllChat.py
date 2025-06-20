from app.db.sql import sql, AsyncSessionLocal
import sqlalchemy as sa


async def get_full_conversation_postgre(chat_id: str):
    async with AsyncSessionLocal() as session:
        # 1. Get chat type and value
        chat_type_result = await session.execute(
            sa.text("SELECT type, value FROM public.chat WHERE chatid = :chatid"),
            {"chatid": chat_id}
        )
        chat_type_row = chat_type_result.fetchone()
        chat_type = chat_type_row.type if chat_type_row else "inconnu"
        chat_value = chat_type_row.value if chat_type_row else "inconnu"

        # 2. Get message history
        query = """
            SELECT content, role, created_at
            FROM message
            WHERE chatid = :chat_id
            ORDER BY created_at ASC
        """
        result = await session.execute(sa.text(query), {"chat_id": chat_id})
        messages = result.fetchall()
        
        if not messages:
            return {
                "type": chat_type,
                "value": chat_value,
                "history": "[Aucune conversation précédente]"
            }
        
        history = []
        for message in messages:
            role = "Utilisateur" if message.role == "user" else "Agent"
            history.append(
                f"{role}: {message.content} (le {message.created_at.strftime('%d %B %Y à %H:%M')})"
            )
        
        return {
            "type": chat_type,
            "value": chat_value,
            "history": "\n".join(history),
        }



