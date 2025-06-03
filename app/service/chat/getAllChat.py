from app.db.sql import sql

async def get_full_conversation_postgre(chat_id: str):
    async with sql() as session:
        query = """
            SELECT message, role, created_at
            FROM chat_messages
            WHERE chat_id = :chat_id
            ORDER BY created_at ASC
        """
        result = await session.execute(query, {"chat_id": chat_id})
        messages = result.fetchall()
        
        if not messages:
            return "[Aucune conversation précédente]"
        
        history = []
        for message in messages:
            role = "Utilisateur" if message.role == "user" else "Agent"
            history.append(f"{role}: {message.message} (le {message.created_at.strftime('%d %B %Y à %H:%M')})")
        
        return "\n".join(history)
     # Test the function with a sample chat_id