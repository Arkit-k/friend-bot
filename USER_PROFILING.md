# User Profiling and Adaptation System

The Friendship Bot includes a sophisticated user profiling and adaptation system that allows it to "read" users and modify its responses based on their communication style, preferences, and interests. This document explains how the system works and how users can interact with it.

## Overview

The user profiling system:

1. **Analyzes user messages** to build a profile of their communication style, topic preferences, and interests/dislikes
2. **Adapts responses** to match the user's communication style and preferences
3. **Remembers interests and dislikes** to personalize conversations
4. **Allows explicit preference setting** through commands

## How It Works

### Communication Style Analysis

The system analyzes several dimensions of a user's communication style:

- **Formality**: How formal or informal the user's language is
- **Verbosity**: Whether the user prefers longer or shorter messages
- **Emotionality**: Whether the user communicates more emotionally or logically
- **Assertiveness**: How assertive or tentative the user's language is
- **Positivity**: How positive or negative the user's messages tend to be
- **Humor**: Whether the user uses humor frequently

### Topic Preference Analysis

The system tracks which topics the user engages with most frequently:

- Personal topics (family, relationships, feelings)
- Professional topics (work, career, business)
- Intellectual topics (books, learning, ideas)
- Recreational topics (games, sports, hobbies)
- Social topics (friends, social events, conversations)
- Emotional topics (feelings, emotions, emotional support)

### Interest and Dislike Detection

The system automatically detects:

- Things the user mentions liking or being interested in
- Things the user mentions disliking or avoiding

### Response Adaptation

Based on the user's profile, the bot adapts its responses in several ways:

1. **Formality Adjustment**: Uses more formal or informal language
2. **Length Adjustment**: Provides longer or shorter responses
3. **Emotional Content**: Adjusts the emotional content of responses
4. **Topic Selection**: Prioritizes topics the user is interested in
5. **Interest References**: Occasionally references the user's interests
6. **Dislike Avoidance**: Avoids mentioning things the user dislikes

## User Commands

Users can interact with the profiling system through these commands:

### View Profile

```
!profile
```

Shows the user's communication profile, including:
- Communication style metrics
- Preferred topics
- Detected interests and dislikes

### Set Preferences

```
!preference [category] [value]
```

Explicitly sets a preference for a specific communication style dimension.

Available categories:
- `formality` (values: formal, neutral, informal)
- `verbosity` (values: verbose, neutral, concise)
- `emotionality` (values: emotional, neutral, logical)
- `positivity` (values: positive, neutral, negative)
- `humor` (values: humorous, neutral, serious)

Example:
```
!preference formality informal
```

### Add Interests and Dislikes

```
!like [something you like]
```

Explicitly tells the bot about something you like or are interested in.

Example:
```
!like chocolate
```

```
!dislike [something you dislike]
```

Explicitly tells the bot about something you dislike or want to avoid.

Example:
```
!dislike spicy food
```

## Privacy and Data Storage

- User profiles are stored locally in the `data/user_profiles` directory
- Each user's profile is stored in a separate JSON file
- Profiles include only information derived from conversations with the bot
- Users can reset their conversation history with the `!reset` command

## Technical Implementation

The user profiling system consists of several components:

1. **UserProfiler** (`user_profiler.py`): Analyzes messages and builds user profiles
2. **ConversationMemory** (`conversation_memory.py`): Stores conversation history and integrates with the profiler
3. **Response Generator** (`response_generator.py`): Adapts responses based on user profiles

The system uses natural language processing techniques to analyze messages, including:
- Keyword detection for topics and interests
- Pattern matching for communication style
- Regular expressions for explicit mentions of likes and dislikes

## Future Improvements

Planned improvements to the user profiling system include:

- More sophisticated natural language understanding
- Better detection of implicit preferences
- More nuanced adaptation of responses
- Learning from user feedback
- More detailed profile visualization
