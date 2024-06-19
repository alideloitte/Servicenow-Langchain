import datetime
from nemoguardrails.actions import action
import re
from typing import Optional

@action(name="greet_based_on_time")
async def greet_based_on_time(
    context: Optional[dict] = None,
):

    greetings = {"hey", "hi", "hallo", "guten", "nice", "hoi"}
    user_response = context.get("last_user_message").lower()
    
    general_intro = "I am a digital assistant for your service." \
                    "how can I help you?"
    
    if any(ext in user_response for ext in greetings):
        current_time = datetime.datetime.now().time()
        if current_time.hour < 12:
            return (
                "Guten Morgen! " + general_intro
            )
        elif current_time.hour < 18:
            return (
                "Guten Tag! " + general_intro
            )
        else:
            return (
                "Guten Abend! " + general_intro
            )
    else:
        return "Hello, I was expecting a joint greeting at first, but don't worry! " + general_intro