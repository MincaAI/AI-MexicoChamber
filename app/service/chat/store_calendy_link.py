from app.db.sql import sql, AsyncSessionLocal
import sqlalchemy as sa

async def store_calendy_link(chat_id: str):
    async with AsyncSessionLocal() as session:
        try:
            async with session.begin():
                # Extract the latest Calendly URL from the `message` table for the given chatid
                result = await session.execute(
                    sa.text("""
                        SELECT regexp_matches(m.content, 'https://calendly.com/[^\\s]+') AS match
                        FROM message m
                        WHERE m.chatid = :chatid AND m.content LIKE '%https://calendly.com/%'
                        ORDER BY m.timestamp DESC
                        LIMIT 1
                    """),
                    {"chatid": chat_id}
                )

                row = result.first()
                if not row or not row[0]:
                    return {"status": "success", "message": "No Calendly link found for this chat"}, 404

                calendy_url = row[0][0]  # Extract the URL string from match array

                # Upsert into meet_clicks
                await session.execute(
                    sa.text("""
                        INSERT INTO meet_clicks (chatid, calendy_url)
                        VALUES (:chatid, :calendy_url)
                        ON CONFLICT (chatid)
                        DO UPDATE SET
                            calendy_url = EXCLUDED.calendy_url,
                            clicked = false
                    """),
                    {
                        "chatid": chat_id,
                        "calendy_url": calendy_url
                    }
                )

            return {"status": "success", "message": "Calendly link upserted successfully"}
        except Exception as e:
            print("Error storing Calendly link:", e)
            return {"status": "error", "message": "Failed to store Calendly link"}, 500