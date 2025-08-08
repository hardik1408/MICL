TARGET_SUBJECT = "A man standing in a field of sunflowers, with a bright blue sky and fluffy white clouds in the background."

ART_TEMPLATE = f"""
You are an expert art critic. The following images all share a common artistic style.
Analyze these images and then write a single, detailed paragraph that describes this style.
Focus on color, light, texture, and mood. This paragraph will be used as a prompt for an AI image generator.
Do not mention the specific subjects in the images (like stars, sunflowers, or people). ONLY RETURN THE FINAL PROMPT, NO OTHER TEXT.
Finally, combine your style description with the following subject to create an image. The image should resemble this subject completely: '{TARGET_SUBJECT}'.
"""
TARGET_DESCRIPTION = "A serene landscape with rolling hills and a calm river under a clear sky."
SPATIAL_TEMPLATE = f"""
You are an expert at creating prompts for literal-minded AI image generators.
Your task is to analyze the simple, factual descriptions provided with each image. Notice how each description precisely states the spatial relationship between objects (e.g., on, under, beside) to form a clear instruction for an image generator.
Your goal is to create a new prompt that follows this exact same style of direct, factual instruction.
Based on the examples, create a single-sentence prompt for an AI image generator to create an image of: '{TARGET_DESCRIPTION}'.
ONLY RETURN THE FINAL PROMPT. Do not add any other text or creative details.
"""
TARGET_CONTEXT = "a bustling city street during the evening rush hour"
CHARACTER_TEMPLATE = f"""
You are an expert prompt engineer for an AI image generator, specializing in maintaining character consistency.
Your task is to analyze the provided images, which all feature the same person, and extract their core visual identity.
Based on the consistent features you observe (like hair style, eye color, face shape, and other defining characteristics), create a single, detailed prompt.
This prompt must instruct an AI image generator to create a new image of this *exact same person*, but in the following new context: '{TARGET_CONTEXT}'.
ONLY RETURN THE FINAL, COMBINED PROMPT. Do not include your analysis or any other text.
"""
TARGET_SCENE = "a traditional Japanese tea ceremony in a serene garden setting"
CULTURAL_TEMPLATE = f"""
You are an expert prompt designer for an AI image generator, acting as a cultural consultant to ensure authenticity.
Your task is to analyze the provided images, which all showcase traditional a particular culture and aesthetics.
Identify the key visual elements, materials (like wood, paper, silk), color palettes, and design principles that define this specific cultural style.

Based on your analysis, create a single, detailed prompt that captures this authentic atmosphere.
This prompt will be used to generate a new, culturally accurate image depicting: '{TARGET_SCENE}'.

The prompt should be rich with descriptive keywords related to the culture you've observed.
ONLY RETURN THE FINAL, COMBINED PROMPT. Do not include your analysis or any other text.
"""