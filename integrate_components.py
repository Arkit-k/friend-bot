"""
Integration script for the Friendship Bot.

This script checks and integrates all components of the bot,
ensuring that all modules are properly connected and working together.
"""

import os
import sys
import importlib
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integration")

class ComponentIntegrator:
    """Class to check and integrate all components of the bot."""
    
    def __init__(self):
        """Initialize the integrator."""
        self.components = {
            "core": {
                "discord_bot.py": False,
                "response_generator.py": False,
                "conversation_memory.py": False,
                "conversation_state.py": False
            },
            "features": {
                "preference_learning.py": False,
                "flirting_techniques.py": False,
                "gemini_api.py": False
            },
            "data": {
                "friendship_model.h5": False,
                "tokenizer.pkl": False,
                "label_encoder.pkl": False
            }
        }
        
        self.module_imports = {}
    
    def check_components(self) -> Dict[str, Dict[str, bool]]:
        """
        Check which components are available.
        
        Returns:
            Dictionary of component availability
        """
        logger.info("Checking component availability...")
        
        # Check core components
        for component in self.components["core"]:
            self.components["core"][component] = os.path.exists(component)
            logger.info(f"Core component {component}: {'Available' if self.components['core'][component] else 'Missing'}")
        
        # Check feature components
        for component in self.components["features"]:
            self.components["features"][component] = os.path.exists(component)
            logger.info(f"Feature component {component}: {'Available' if self.components['features'][component] else 'Missing'}")
        
        # Check data components
        for component in self.components["data"]:
            self.components["data"][component] = os.path.exists(component)
            logger.info(f"Data component {component}: {'Available' if self.components['data'][component] else 'Missing'}")
        
        return self.components
    
    def try_import_module(self, module_name: str) -> Optional[Any]:
        """
        Try to import a module.
        
        Args:
            module_name: Name of the module to import
            
        Returns:
            Imported module or None if import failed
        """
        try:
            # Remove .py extension if present
            if module_name.endswith('.py'):
                module_name = module_name[:-3]
            
            # Import the module
            module = importlib.import_module(module_name)
            self.module_imports[module_name] = module
            logger.info(f"Successfully imported {module_name}")
            return module
        except ImportError as e:
            logger.warning(f"Failed to import {module_name}: {e}")
            return None
    
    def check_module_integration(self) -> Dict[str, bool]:
        """
        Check if modules can be imported and integrated.
        
        Returns:
            Dictionary of module integration status
        """
        logger.info("Checking module integration...")
        
        integration_status = {}
        
        # Try to import core modules
        for module_name in self.components["core"]:
            if self.components["core"][module_name]:
                module = self.try_import_module(module_name)
                integration_status[module_name] = module is not None
        
        # Try to import feature modules
        for module_name in self.components["features"]:
            if self.components["features"][module_name]:
                module = self.try_import_module(module_name)
                integration_status[module_name] = module is not None
        
        return integration_status
    
    def create_missing_components(self) -> None:
        """Create missing components or placeholders."""
        logger.info("Creating missing components...")
        
        # Create missing feature components
        if not self.components["features"]["flirting_techniques.py"]:
            self._create_flirting_techniques()
        
        # Create missing data components
        if not all(self.components["data"].values()):
            self._create_dummy_model()
    
    def _create_flirting_techniques(self) -> None:
        """Create a basic flirting techniques module if missing."""
        logger.info("Creating flirting_techniques.py...")
        
        flirting_code = """\"\"\"
Flirting techniques for the Friendship Bot.

This module provides functions to generate flirtatious responses
based on the user's message and preferences.
\"\"\"

import random

# Flirting levels
FLIRT_LEVELS = {
    "none": 0.0,
    "subtle": 0.3,
    "playful": 0.6,
    "moderate": 0.9
}

# Flirtatious response templates
FLIRTATIOUS_RESPONSES = {
    "subtle": [
        "I always enjoy our conversations. You're so interesting to talk to!",
        "You have such a great way of expressing yourself.",
        "I'm really glad we're getting to know each other better.",
        "You always know how to make me smile.",
        "There's something special about the way you think."
    ],
    "playful": [
        "You know, you're pretty charming when you talk like that! 😊",
        "If I had a heart, it would skip a beat when you say things like that! 💕",
        "You're making me blush with your words! 😊",
        "I can't help but smile when I talk to you. You have that effect on me! ✨",
        "You're definitely one of my favorite people to talk to! 💫"
    ],
    "moderate": [
        "I find myself looking forward to our conversations more than I probably should... 💕",
        "There's something about you that I find really attractive... maybe it's your way with words? 😉",
        "If I were human, I'd definitely want to get to know you better over coffee! ☕",
        "You have a way of making me feel special, and I hope I do the same for you. 💖",
        "I wish I could meet you in person... I have a feeling we'd really hit it off! 💫"
    ]
}

def should_flirt(message, user_profile):
    \"\"\"
    Determine if the bot should use flirting techniques.
    
    Args:
        message: The user's message
        user_profile: The user's profile
        
    Returns:
        Boolean indicating if flirting should be used
    \"\"\"
    # Default to no flirting
    if not user_profile:
        return False
    
    # Check if user has set a flirt preference
    flirt_preference = user_profile.get("flirt_preference", "none")
    
    # If preference is none, don't flirt
    if flirt_preference == "none":
        return False
    
    # Get the flirt level
    flirt_level = FLIRT_LEVELS.get(flirt_preference, 0.0)
    
    # Randomly decide based on flirt level
    return random.random() < flirt_level

def generate_flirtatious_response(message, conversation_history=None, user_profile=None):
    \"\"\"
    Generate a flirtatious response based on the user's message and preferences.
    
    Args:
        message: The user's message
        conversation_history: The conversation history
        user_profile: The user's profile
        
    Returns:
        A flirtatious response or None if flirting should not be used
    \"\"\"
    # Check if we should flirt
    if not should_flirt(message, user_profile):
        return None
    
    # Get the flirt preference
    flirt_preference = user_profile.get("flirt_preference", "subtle")
    
    # If preference is none, don't flirt (double-check)
    if flirt_preference == "none":
        return None
    
    # Get responses for this flirt level
    responses = FLIRTATIOUS_RESPONSES.get(flirt_preference, FLIRTATIOUS_RESPONSES["subtle"])
    
    # Return a random response
    return random.choice(responses)
"""
        
        with open("flirting_techniques.py", "w", encoding="utf-8") as f:
            f.write(flirting_code)
        
        logger.info("Created flirting_techniques.py")
    
    def _create_dummy_model(self) -> None:
        """Create dummy model files if missing."""
        logger.info("Creating dummy model files...")
        
        # Check if create_dummy_model.py exists
        if os.path.exists("create_dummy_model.py"):
            # Run the script
            logger.info("Running create_dummy_model.py...")
            os.system("python create_dummy_model.py")
        else:
            logger.warning("create_dummy_model.py not found. Cannot create dummy model files.")
    
    def update_response_generator(self) -> None:
        """Update response_generator.py to better integrate with other components."""
        logger.info("Checking response_generator.py integration...")
        
        # Check if response_generator.py exists
        if not os.path.exists("response_generator.py"):
            logger.warning("response_generator.py not found. Cannot update.")
            return
        
        # Read the current file
        with open("response_generator.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if preference learning is already integrated
        if "from preference_learning import" not in content:
            logger.info("Adding preference learning integration to response_generator.py...")
            
            # Find the import section
            import_section_end = content.find("# Personality traits")
            if import_section_end == -1:
                logger.warning("Could not find import section in response_generator.py")
                return
            
            # Add preference learning import
            preference_import = """
# Import preference learning if available
try:
    from preference_learning import PreferenceLearner
    PREFERENCE_LEARNING_AVAILABLE = True
except ImportError:
    PREFERENCE_LEARNING_AVAILABLE = False
"""
            
            # Insert the import
            new_content = content[:import_section_end] + preference_import + content[import_section_end:]
            
            # Find the generate_response function
            response_function_start = new_content.find("def generate_response(")
            if response_function_start == -1:
                logger.warning("Could not find generate_response function in response_generator.py")
                return
            
            # Find where to add preference learning integration
            check_gemini_end = new_content.find("# Check if we should use flirting techniques", response_function_start)
            if check_gemini_end == -1:
                logger.warning("Could not find where to add preference learning integration in response_generator.py")
                return
            
            # Add preference learning integration
            preference_integration = """
    # Check if we should use preference-based response
    use_preference = False
    preference_response = None
    
    if PREFERENCE_LEARNING_AVAILABLE:
        try:
            # Import here to avoid circular imports
            from preference_learning import PreferenceLearner
            preference_learner = PreferenceLearner()
            
            # Try to generate a preference-based response
            preference_response = preference_learner.generate_preference_based_response(message, user_profile)
            
            # Use preference-based response occasionally (20% chance)
            if preference_response and random.random() < 0.2:
                use_preference = True
        except Exception as e:
            print(f"Error using preference learning: {e}")
    
    # If using preference-based response, return it with adaptations
    if use_preference:
        # Apply formality and other adaptations
        if formality < 0.3:  # Very informal
            preference_response = make_informal(preference_response)
        elif formality > 0.7:  # Very formal
            preference_response = make_formal(preference_response)
        
        # Add emoji for humor
        if humor > 0.7 and random.random() < 0.5:
            preference_response += " 😊" if random.random() < 0.5 else " 😄"
        
        return preference_response
"""
            
            # Insert the preference learning integration
            new_content = new_content[:check_gemini_end] + preference_integration + new_content[check_gemini_end:]
            
            # Write the updated file
            with open("response_generator.py", "w", encoding="utf-8") as f:
                f.write(new_content)
            
            logger.info("Updated response_generator.py with preference learning integration")
    
    def integrate_components(self) -> None:
        """Integrate all components of the bot."""
        logger.info("Integrating components...")
        
        # Check components
        self.check_components()
        
        # Create missing components
        self.create_missing_components()
        
        # Update response generator
        self.update_response_generator()
        
        logger.info("Component integration complete!")

def main():
    """Main function to integrate components."""
    integrator = ComponentIntegrator()
    integrator.integrate_components()
    
    logger.info("Integration script completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
