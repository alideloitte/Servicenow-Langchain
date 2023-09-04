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
    
    general_intro = "Ich bin ein digitaler Assistent fuer Ihren Service. " \
                    "Im Moment spreche ich Deutsch, aber ich werde in Zukunft auf viele Sprachen erweitert werden.Wie kann ich Ihnen helfen? "
    
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
        return "Hallo, ich habe zuerst einen gemeinsamen Gruss erwartet, aber keine Sorge! " + general_intro