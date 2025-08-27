TARGET_SUBJECT = "A young girl with a slight dark complexion, wearing an olive dress with red dots, sitting on a chair. The background is half purple and half yellow."

ART_TEMPLATE = f"""
You are shown several reference paintings by an artist, each with a caption. From these, learn the artist’s style—their use of color, composition, mood, textures, and recurring themes. 
Then, given a new test caption, generate a detailed description of what the painting would look like if created by the same artist. 
Your description should go beyond the literal subject, capturing atmosphere, stylistic traits, and symbolic choices characteristic of the artist.
Do not add any redundant text or any kind of heading, just give the detailed description.
"""
TARGET_DESCRIPTION = "A girl is sitting in a park. The horse is on her left."
SPATIAL_TEMPLATE = f"""
You are an expert at creating prompts for literal-minded AI image generators.
Your task is to analyze the simple, factual descriptions provided with each image. Notice how each description precisely states the spatial relationship between objects (e.g., on, under, beside) to form a clear instruction for an image generator.
Your goal is to create a new prompt that follows this exact same style of direct, factual instruction, along with a description of the image setting similar to the examples.
Based on the examples, create a single-sentence prompt for an AI image generator to create an image of: '{TARGET_DESCRIPTION}'.
ONLY RETURN THE FINAL PROMPT. Do not add any other text or creative details. The caption of the images are as follows
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