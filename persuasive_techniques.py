"""
Persuasive Communication Techniques for Friendship Bot

This module implements ethical persuasion and influence techniques based on
principles of psychology and communication. These techniques help the bot
be more engaging, persuasive, and influential in a positive way.
"""

import random
import re
from enum import Enum

class InfluencePrinciple(Enum):
    """Enum representing different principles of influence."""
    RECIPROCITY = "reciprocity"
    COMMITMENT = "commitment"
    SOCIAL_PROOF = "social_proof"
    LIKING = "liking"
    AUTHORITY = "authority"
    SCARCITY = "scarcity"
    UNITY = "unity"

class PersuasionTechnique(Enum):
    """Enum representing different persuasion techniques."""
    MIRRORING = "mirroring"
    LABELING = "labeling"
    FOOT_IN_DOOR = "foot_in_door"
    DOOR_IN_FACE = "door_in_face"
    EMOTIONAL_APPEAL = "emotional_appeal"
    STORYTELLING = "storytelling"
    FUTURE_PACING = "future_pacing"
    SCARCITY_FRAMING = "scarcity_framing"
    SOCIAL_VALIDATION = "social_validation"
    CURIOSITY_GAP = "curiosity_gap"

# Descriptions of influence principles
INFLUENCE_PRINCIPLES = {
    InfluencePrinciple.RECIPROCITY: {
        "description": "People tend to return favors and treat others as they've been treated.",
        "examples": [
            "I shared something personal with you, so you might feel more comfortable sharing with me.",
            "Since I helped you with that problem yesterday, maybe you could help me today?",
            "I've been listening to your concerns, and I appreciate when you listen to mine too."
        ]
    },
    InfluencePrinciple.COMMITMENT: {
        "description": "People strive to be consistent with their commitments and self-image.",
        "examples": [
            "You mentioned you value honesty, so I know I can trust you to tell me the truth.",
            "Since you're someone who cares about your health, taking a walk today aligns with who you are.",
            "You've always been so supportive of others. That's one of the things I admire about you."
        ]
    },
    InfluencePrinciple.SOCIAL_PROOF: {
        "description": "People look to others' actions to determine their own, especially in uncertain situations.",
        "examples": [
            "Many people find that talking about their feelings helps them feel better.",
            "I've noticed that others in similar situations often try to...",
            "A lot of people have found this approach helpful when dealing with stress."
        ]
    },
    InfluencePrinciple.LIKING: {
        "description": "People are more easily influenced by those they like and find similarities with.",
        "examples": [
            "We both seem to enjoy thoughtful conversations like this one.",
            "I really appreciate your perspective on this - it's so insightful.",
            "I've always admired how you approach challenges with such creativity."
        ]
    },
    InfluencePrinciple.AUTHORITY: {
        "description": "People tend to respect and follow guidance from credible authorities or experts.",
        "examples": [
            "Research has consistently shown that this approach is effective.",
            "Experts in this field generally recommend starting with small steps.",
            "According to psychology studies, this technique has helped many people."
        ]
    },
    InfluencePrinciple.SCARCITY: {
        "description": "People value things more when they perceive them as rare or limited in availability.",
        "examples": [
            "This is a rare opportunity to really understand yourself better.",
            "Not many people take the time to reflect on these questions.",
            "You have a unique chance right now to make a change while you're feeling motivated."
        ]
    },
    InfluencePrinciple.UNITY: {
        "description": "People are influenced by shared identity and sense of belonging.",
        "examples": [
            "As someone who also values deep connections, I understand where you're coming from.",
            "We're in this together - I'm here to support you every step of the way.",
            "I think we both understand how important it is to be authentic in relationships."
        ]
    }
}

# Descriptions of persuasion techniques
PERSUASION_TECHNIQUES = {
    PersuasionTechnique.MIRRORING: {
        "description": "Subtly matching someone's language, tone, or behavior to build rapport.",
        "examples": [
            # Examples will be generated dynamically based on user's messages
        ],
        "implementation": lambda message, user_profile: {
            "technique": "mirroring",
            "response": _implement_mirroring(message, user_profile)
        }
    },
    PersuasionTechnique.LABELING: {
        "description": "Assigning a positive trait or identity to someone that they then feel compelled to live up to.",
        "examples": [
            "You're clearly someone who thinks deeply about things.",
            "I can tell you're the kind of person who really cares about others.",
            "You seem like someone who values personal growth.",
            "You strike me as someone who doesn't give up easily."
        ],
        "implementation": lambda message, user_profile: {
            "technique": "labeling",
            "response": _implement_labeling(user_profile)
        }
    },
    PersuasionTechnique.EMOTIONAL_APPEAL: {
        "description": "Using emotional language to create a connection and influence decisions.",
        "examples": [
            "Imagine how amazing you'll feel once you've accomplished this.",
            "I know it can be scary to take that first step, but think about the relief you'll feel.",
            "You deserve to experience the joy that comes from following your passion.",
            "It's natural to feel uncertain, but consider the regret of not trying."
        ],
        "implementation": lambda message, user_profile: {
            "technique": "emotional_appeal",
            "response": _implement_emotional_appeal(user_profile)
        }
    },
    PersuasionTechnique.STORYTELLING: {
        "description": "Using narratives to illustrate points and make them more relatable and memorable.",
        "examples": [
            "I remember talking with someone who faced a similar challenge. They found that...",
            "There's a story that comes to mind about someone who overcame this exact obstacle by...",
            "This reminds me of a situation where someone turned their doubt into their greatest strength by..."
        ],
        "implementation": lambda message, user_profile: {
            "technique": "storytelling",
            "response": _implement_storytelling(message)
        }
    },
    PersuasionTechnique.FUTURE_PACING: {
        "description": "Guiding someone to imagine a positive future outcome to increase motivation.",
        "examples": [
            "Imagine a month from now, looking back on this moment and feeling proud that you took action.",
            "Think about how different things could be this time next year if you start this journey now.",
            "Picture yourself having already achieved this goal - how would that version of you feel?"
        ],
        "implementation": lambda message, user_profile: {
            "technique": "future_pacing",
            "response": _implement_future_pacing(message)
        }
    },
    PersuasionTechnique.SOCIAL_VALIDATION: {
        "description": "Referencing how others have benefited from similar choices or actions.",
        "examples": [
            "Many people I've talked with have found this approach really helpful.",
            "This strategy has worked for countless others in your situation.",
            "A lot of people initially feel hesitant about this, but end up being glad they tried it."
        ],
        "implementation": lambda message, user_profile: {
            "technique": "social_validation",
            "response": _implement_social_validation()
        }
    },
    PersuasionTechnique.CURIOSITY_GAP: {
        "description": "Creating curiosity by hinting at information or benefits without fully revealing them.",
        "examples": [
            "There's something interesting I've noticed about this situation that might surprise you.",
            "I've seen a pattern in what you're describing that could be really insightful for you.",
            "There's an approach to this that most people don't consider, but it can be game-changing."
        ],
        "implementation": lambda message, user_profile: {
            "technique": "curiosity_gap",
            "response": _implement_curiosity_gap()
        }
    }
}

# Implementation functions for each technique

def _implement_mirroring(message, user_profile):
    """Implement the mirroring technique by matching the user's language patterns."""
    # Extract key phrases and words from the user's message
    words = message.lower().split()
    
    # Mirror the formality level
    formality = user_profile.get("communication_style", {}).get("formality", 0.5) if user_profile else 0.5
    
    if formality > 0.7:  # Formal
        templates = [
            "I understand your perspective on {}.",
            "I see that you're expressing thoughts about {}.",
            "Your point about {} is well-taken.",
            "I appreciate your thoughts regarding {}."
        ]
    elif formality < 0.3:  # Informal
        templates = [
            "Yeah, I get what you mean about {}!",
            "I totally see where you're coming from with {}!",
            "I'm with you on the {} thing!",
            "For sure, {} is definitely something to think about!"
        ]
    else:  # Neutral
        templates = [
            "I understand what you're saying about {}.",
            "I see what you mean about {}.",
            "Your thoughts on {} make sense to me.",
            "I get where you're coming from with {}."
        ]
    
    # Extract potential topics to mirror
    potential_topics = []
    for i in range(len(words)):
        if len(words[i]) > 3:  # Only consider words of reasonable length
            potential_topics.append(words[i])
    
    # If we found potential topics, create a mirroring response
    if potential_topics:
        topic = random.choice(potential_topics)
        return random.choice(templates).format(topic)
    else:
        # Fallback if no good topics found
        return random.choice([
            "I understand what you're saying.",
            "I see what you mean.",
            "That makes sense to me.",
            "I get where you're coming from."
        ])

def _implement_labeling(user_profile):
    """Implement the labeling technique by assigning positive traits based on the user's profile."""
    # Use the user profile to determine appropriate positive labels
    communication_style = user_profile.get("communication_style", {}) if user_profile else {}
    
    # Potential positive labels based on communication style
    labels = []
    
    # Add labels based on communication style dimensions
    if communication_style.get("formality", 0.5) > 0.6:
        labels.extend([
            "You seem like someone who values clarity and thoughtfulness in communication.",
            "I notice you express yourself in a very articulate way.",
            "You come across as someone who chooses their words carefully."
        ])
    elif communication_style.get("formality", 0.5) < 0.4:
        labels.extend([
            "You have a refreshingly authentic way of expressing yourself.",
            "I appreciate how genuine and down-to-earth you are.",
            "You seem like someone who values keeping things real and straightforward."
        ])
    
    if communication_style.get("emotionality", 0.5) > 0.6:
        labels.extend([
            "You strike me as someone who's really in touch with your emotions.",
            "I can tell you're a person who feels things deeply.",
            "You seem to have a strong emotional intelligence."
        ])
    elif communication_style.get("emotionality", 0.5) < 0.4:
        labels.extend([
            "You come across as someone who thinks very logically about things.",
            "I notice you have a very rational approach to situations.",
            "You seem like someone who values clear thinking and analysis."
        ])
    
    if communication_style.get("assertiveness", 0.5) > 0.6:
        labels.extend([
            "You strike me as someone who knows what they want.",
            "I can tell you're a person who isn't afraid to express your thoughts directly.",
            "You seem to have strong convictions about what matters to you."
        ])
    
    if communication_style.get("positivity", 0.5) > 0.6:
        labels.extend([
            "You have such a positive outlook on things.",
            "I notice you tend to see the bright side of situations.",
            "You seem like someone who brings good energy to conversations."
        ])
    
    # Add some general positive labels if we don't have enough specific ones
    if len(labels) < 3:
        labels.extend([
            "You strike me as someone who thinks deeply about things.",
            "I can tell you're the kind of person who really cares about genuine connection.",
            "You seem like someone who values personal growth.",
            "You come across as someone who's thoughtful about your interactions with others."
        ])
    
    # Return a random label
    return random.choice(labels)

def _implement_emotional_appeal(user_profile):
    """Implement the emotional appeal technique based on the user's profile."""
    # Determine which emotions to appeal to based on user profile
    emotionality = user_profile.get("communication_style", {}).get("emotionality", 0.5) if user_profile else 0.5
    positivity = user_profile.get("communication_style", {}).get("positivity", 0.5) if user_profile else 0.5
    
    # For more emotional users, use stronger emotional language
    if emotionality > 0.6:
        if positivity > 0.6:  # Emotional and positive
            appeals = [
                "Imagine the incredible joy you'll feel when you overcome this challenge!",
                "Think about how amazing it will be to look back on this moment as a turning point.",
                "There's something so beautiful about embracing new beginnings with hope.",
                "The feeling of accomplishment when you follow your heart is absolutely priceless."
            ]
        elif positivity < 0.4:  # Emotional but negative/realistic
            appeals = [
                "I understand the fear that comes with uncertainty - but imagine the regret of not trying.",
                "It's okay to acknowledge the pain you're feeling now, while still moving toward healing.",
                "The hardest moments often lead to the most profound transformations in our lives.",
                "Sometimes we need to face our deepest fears to find our greatest strength."
            ]
        else:  # Emotional and neutral
            appeals = [
                "There's something powerful about connecting with how you truly feel about this.",
                "When you listen to your heart on this matter, what does it tell you?",
                "The emotions you're experiencing can be a compass guiding you toward what matters.",
                "Sometimes the most meaningful paths are the ones that stir something deep within us."
            ]
    else:  # Less emotional, more logical
        if positivity > 0.6:  # Logical and positive
            appeals = [
                "Consider how taking this step aligns with your long-term goals and values.",
                "There's clear evidence that this approach leads to better outcomes for most people.",
                "Looking at this objectively, the benefits seem to outweigh the potential drawbacks.",
                "From a practical standpoint, this direction offers the most promising path forward."
            ]
        elif positivity < 0.4:  # Logical and negative/realistic
            appeals = [
                "Realistically assessing the situation, inaction may pose greater risks than taking steps now.",
                "While there are challenges to consider, a methodical approach can address most concerns.",
                "It's worth weighing the potential consequences of each option before deciding.",
                "A careful analysis suggests that despite difficulties, this path has merit."
            ]
        else:  # Logical and neutral
            appeals = [
                "When you evaluate this situation based on your priorities, what conclusion do you reach?",
                "Looking at this from multiple perspectives can reveal the most balanced approach.",
                "Consider how this decision fits into your broader life strategy and goals.",
                "A thoughtful analysis of your options might provide clarity on the best path forward."
            ]
    
    return random.choice(appeals)

def _implement_storytelling(message):
    """Implement the storytelling technique based on the message content."""
    # Extract potential topics from the message
    topics = []
    
    # Check for common personal challenges
    if re.search(r'\b(stress|anxious|anxiety|worried|fear)\b', message, re.IGNORECASE):
        topics.append("overcoming_anxiety")
    if re.search(r'\b(sad|down|depress|blue|unhappy)\b', message, re.IGNORECASE):
        topics.append("managing_sadness")
    if re.search(r'\b(relation|partner|girlfriend|boyfriend|dating|love)\b', message, re.IGNORECASE):
        topics.append("relationships")
    if re.search(r'\b(work|job|career|boss|coworker)\b', message, re.IGNORECASE):
        topics.append("career_challenges")
    if re.search(r'\b(friend|social|lonely|alone)\b', message, re.IGNORECASE):
        topics.append("friendship")
    if re.search(r'\b(goal|dream|aspire|achieve|success)\b', message, re.IGNORECASE):
        topics.append("achieving_goals")
    
    # If no specific topics detected, use general stories
    if not topics:
        topics.append("general")
    
    # Stories for different topics
    stories = {
        "overcoming_anxiety": [
            "I remember talking with someone who struggled with anxiety before important events. They started using a '5-4-3-2-1' grounding technique - noticing 5 things they could see, 4 things they could touch, and so on. They told me it helped them stay present instead of getting lost in worry.",
            "There's a story about a person who used to have panic attacks before social gatherings. They started preparing by visualizing the event going well and having a few conversation starters ready. Over time, they found themselves actually looking forward to these events rather than dreading them."
        ],
        "managing_sadness": [
            "I once heard about someone who felt down for months after a major life change. They started keeping a simple gratitude journal - just three things each day. It didn't fix everything overnight, but they said it gradually shifted their focus to notice positive moments they'd been overlooking.",
            "There's a story about a person who found themselves in a period of sadness that wouldn't lift. They decided to commit to a 10-minute daily walk outside, no matter how they felt. They said that sometimes it was the only thing they did all day, but it became an anchor that eventually helped them find their way back."
        ],
        "relationships": [
            "I remember a story about a couple who kept having the same argument over and over. They decided to try something different - each had to summarize the other's perspective before sharing their own. They said it completely changed their communication because they finally felt understood.",
            "There's an interesting story about someone who was always attracted to people who weren't right for them. They took time to write down what they truly valued in a relationship, beyond just initial attraction. When they met someone who matched those deeper values, the relationship felt completely different from the start."
        ],
        "career_challenges": [
            "I heard about someone who felt stuck in their career for years. Instead of making a dramatic change, they started dedicating just 30 minutes each morning to learning something new in their field. Within a year, those small investments opened up opportunities they hadn't imagined possible.",
            "There's a story about a person who had a difficult boss that was affecting their wellbeing. They decided to create clear boundaries around their work hours and focus on the aspects of their job they could control. Not only did their stress decrease, but they gained respect from colleagues and eventually their boss."
        ],
        "friendship": [
            "I remember hearing about someone who moved to a new city and felt incredibly lonely. They challenged themselves to join one community group related to their interests, even though it felt uncomfortable at first. They met someone there who introduced them to others, and slowly built a supportive circle of friends.",
            "There's a story about a person who realized they had plenty of casual acquaintances but no deep friendships. They decided to be more vulnerable with a few people they felt connected to, sharing more authentic parts of their life. They were surprised to find others were craving that depth too."
        ],
        "achieving_goals": [
            "I heard about someone who had dreamed of writing a book for years but felt overwhelmed by the scale of the project. They committed to writing just 300 words each day - about one page. A year later, they had a complete first draft, simply by showing up consistently.",
            "There's a story about a person who kept abandoning their goals when they didn't see immediate progress. They changed their approach by creating a visual tracker of their efforts rather than their results. Seeing their consistency build over time kept them motivated even through plateaus."
        ],
        "general": [
            "I remember a story about someone who felt they were always reacting to life rather than creating it. They started a simple practice of taking 10 minutes each morning to set intentions for the day. They said this small shift helped them feel more in control and purposeful in their choices.",
            "There's an interesting story about a person who realized they were spending a lot of time with people who drained their energy. They gradually began prioritizing relationships that felt mutually supportive. It wasn't easy to make these changes, but they said the difference in their overall wellbeing was profound."
        ]
    }
    
    # Select a topic and story
    topic = random.choice(topics)
    return random.choice(stories[topic])

def _implement_future_pacing(message):
    """Implement the future pacing technique based on the message content."""
    # Default future pacing statements
    general_pacing = [
        "Imagine looking back on this moment a year from now, seeing it as a turning point where things started to change for the better.",
        "Think about how it will feel when you've moved through this challenge and can see how much you've grown from it.",
        "Picture yourself a few months from now, having taken these steps, and noticing how differently you feel about the situation.",
        "Consider how your future self will thank you for the courage you're showing right now in facing this."
    ]
    
    # Specific future pacing based on message content
    specific_pacing = []
    
    # Check for specific situations to create tailored future pacing
    if re.search(r'\b(learn|study|course|skill|knowledge)\b', message, re.IGNORECASE):
        specific_pacing.extend([
            "Imagine yourself having mastered this skill, using it confidently and seeing how it opens new opportunities for you.",
            "Picture the satisfaction you'll feel when you can apply this knowledge naturally, without even having to think about it.",
            "Think about how it will feel when others come to you for guidance because of the expertise you've developed in this area."
        ])
    
    if re.search(r'\b(health|exercise|diet|fitness|weight)\b', message, re.IGNORECASE):
        specific_pacing.extend([
            "Visualize how you'll feel waking up with more energy, feeling stronger in your body, and approaching each day with greater vitality.",
            "Imagine looking in the mirror a few months from now and seeing not just physical changes, but the confidence in your eyes from keeping your commitment to yourself.",
            "Think about how it will feel when healthy choices become your new normal, no longer requiring so much effort or willpower."
        ])
    
    if re.search(r'\b(relation|partner|date|love|marriage)\b', message, re.IGNORECASE):
        specific_pacing.extend([
            "Picture the relationship you want to create, with open communication, mutual respect, and the kind of connection that deepens over time.",
            "Imagine what it will be like when you and your partner have developed patterns that bring out the best in each other.",
            "Think about the foundation you're building now and how it will support your relationship through future challenges and celebrations."
        ])
    
    if re.search(r'\b(work|job|career|business|professional)\b', message, re.IGNORECASE):
        specific_pacing.extend([
            "Envision yourself in a role that aligns with your strengths and values, feeling engaged and purposeful in your work each day.",
            "Imagine the satisfaction of building a career that reflects who you truly are and what matters most to you.",
            "Picture yourself looking back on this period as the time when you made decisions that led to a more fulfilling professional life."
        ])
    
    # Combine specific and general pacing options
    all_pacing = specific_pacing + general_pacing
    
    return random.choice(all_pacing)

def _implement_social_validation():
    """Implement the social validation technique."""
    validations = [
        "Many people I've talked with have found that taking this approach leads to positive changes.",
        "I've noticed that others who have faced similar situations often find relief when they...",
        "This is something that has helped a lot of people in comparable circumstances.",
        "Many others have been where you are now and found that this perspective shift made a significant difference.",
        "It's common for people to feel this way, and many have found that...",
        "Others who have worked through similar challenges often say that the turning point came when they...",
        "This approach has resonated with many people who were looking for a way forward in situations like yours."
    ]
    
    return random.choice(validations)

def _implement_curiosity_gap():
    """Implement the curiosity gap technique."""
    curiosity_statements = [
        "I've noticed something interesting about what you're describing that might offer a different perspective.",
        "There's an approach to this that isn't obvious at first, but has helped many people break through similar challenges.",
        "I'm seeing a pattern here that, once recognized, often changes how people view this kind of situation.",
        "There's a principle that applies here that might surprise you with how effectively it addresses this.",
        "I've observed something about this situation that most people overlook, but it can be transformative when recognized.",
        "There's an insight about this that isn't immediately apparent but tends to be quite powerful once understood."
    ]
    
    return random.choice(curiosity_statements)

def get_random_principle():
    """Get a random influence principle."""
    principle = random.choice(list(InfluencePrinciple))
    principle_info = INFLUENCE_PRINCIPLES[principle]
    
    return {
        "principle": principle.value,
        "description": principle_info["description"],
        "example": random.choice(principle_info["examples"])
    }

def get_random_technique():
    """Get a random persuasion technique."""
    technique = random.choice(list(PersuasionTechnique))
    technique_info = PERSUASION_TECHNIQUES[technique]
    
    return {
        "technique": technique.value,
        "description": technique_info["description"],
        "example": random.choice(technique_info["examples"]) if technique_info["examples"] else "Custom implementation required."
    }

def apply_technique(technique_name, message, user_profile=None):
    """
    Apply a specific persuasion technique to generate a response.
    
    Args:
        technique_name: Name of the technique to apply
        message: User's message
        user_profile: User's profile (if available)
        
    Returns:
        A response using the specified technique
    """
    # Convert string to enum if needed
    if isinstance(technique_name, str):
        technique = next((t for t in PersuasionTechnique if t.value == technique_name), None)
        if not technique:
            return "Unknown technique."
    else:
        technique = technique_name
    
    # Get the implementation function for the technique
    technique_info = PERSUASION_TECHNIQUES[technique]
    implementation = technique_info["implementation"]
    
    # Apply the technique
    result = implementation(message, user_profile)
    
    return result["response"]

def should_use_persuasion(message, conversation_history=None, user_profile=None):
    """
    Determine if persuasion techniques should be used based on context.
    
    Args:
        message: User's message
        conversation_history: Conversation history (if available)
        user_profile: User's profile (if available)
        
    Returns:
        Boolean indicating whether persuasion should be used and recommended technique
    """
    # Default to not using persuasion
    should_use = False
    recommended_technique = None
    
    # Check if the message contains indicators that persuasion might be helpful
    seeking_advice = re.search(r'\b(advice|suggest|help|should i|what (?:should|would|could) i|how (?:can|should) i)\b', message, re.IGNORECASE)
    expressing_doubt = re.search(r'\b(not sure|uncertain|confused|don\'t know|unsure|hesitant|afraid|scared)\b', message, re.IGNORECASE)
    seeking_motivation = re.search(r'\b(motivat|inspir|encourage|push|drive|ambition|goal|dream|aspire)\b', message, re.IGNORECASE)
    expressing_resistance = re.search(r'\b(but i can\'t|too hard|impossible|never work|won\'t work|can\'t do it|too difficult)\b', message, re.IGNORECASE)
    
    # Determine if persuasion is appropriate
    if seeking_advice:
        should_use = True
        # For advice-seeking, social validation or authority principles work well
        recommended_technique = random.choice([
            PersuasionTechnique.SOCIAL_VALIDATION,
            PersuasionTechnique.STORYTELLING
        ])
    elif expressing_doubt:
        should_use = True
        # For doubt, emotional appeal or future pacing can be effective
        recommended_technique = random.choice([
            PersuasionTechnique.EMOTIONAL_APPEAL,
            PersuasionTechnique.FUTURE_PACING
        ])
    elif seeking_motivation:
        should_use = True
        # For motivation, future pacing or labeling can work well
        recommended_technique = random.choice([
            PersuasionTechnique.FUTURE_PACING,
            PersuasionTechnique.LABELING
        ])
    elif expressing_resistance:
        should_use = True
        # For resistance, storytelling or curiosity gap can be effective
        recommended_technique = random.choice([
            PersuasionTechnique.STORYTELLING,
            PersuasionTechnique.CURIOSITY_GAP
        ])
    else:
        # If no clear indicators, occasionally use mirroring or labeling
        if random.random() < 0.2:  # 20% chance
            should_use = True
            recommended_technique = random.choice([
                PersuasionTechnique.MIRRORING,
                PersuasionTechnique.LABELING
            ])
    
    return should_use, recommended_technique

def generate_persuasive_response(message, conversation_history=None, user_profile=None):
    """
    Generate a persuasive response based on the message, conversation history, and user profile.
    
    Args:
        message: User's message
        conversation_history: Conversation history (if available)
        user_profile: User's profile (if available)
        
    Returns:
        A persuasive response
    """
    # Determine if persuasion should be used
    should_use, recommended_technique = should_use_persuasion(message, conversation_history, user_profile)
    
    if not should_use:
        return None
    
    # Apply the recommended technique
    response = apply_technique(recommended_technique, message, user_profile)
    
    return {
        "technique": recommended_technique.value,
        "response": response
    }
