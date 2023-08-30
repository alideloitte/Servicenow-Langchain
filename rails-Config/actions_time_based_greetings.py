import datetime
from nemoguardrails.actions import action
import re
from typing import Optional


@action(name="greet_based_on_time")
async def greet_based_on_time(
    context: Optional[dict] = None,
):

    greetings = {"hey", "hi", "hello", "good", "nice", "ho", "huhu"}
    user_response = context.get("last_user_message").lower()
    
    general_intro = "I am a digital assistant for your new ID.4. " \
                    "I will help you search the owner's manual. " \
                    "Please note, that I am not designed to help you with any troubleshooting " \
                    "but rather to answer questions about the content of the owner's manual.  " \
                    "For now I speak british English but will be extended to many languages in the future. " \
                    "May I ask your name?"
    
    if any(ext in user_response for ext in greetings):
        current_time = datetime.datetime.now().time()
        if current_time.hour < 12:
            return (
                "Good morning! " + general_intro
            )
        elif current_time.hour < 18:
            return (
                "Good afternoon! " + general_intro
            )
        else:
            return (
                "Good evening! " + general_intro
            )
    else:
        return "Hi there, I expected a common greeting first but no worries! " + general_intro