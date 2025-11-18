
import io
import random

from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw


def minor_crop(img, min_scale=0.9, max_scale=0.95):
    #Randomly crop a small border from the image and resize back to original size.
    
    w, h = img.size
    scale = random.uniform(min_scale, max_scale) 
    new_w, new_h = int(w * scale), int(h * scale)

    if new_w <= 0 or new_h <= 0: # if the new width or height is less than or equal to 0, return the image
        return img

    left = random.randint(0, w - new_w) # generate a random left coordinate
    top = random.randint(0, h - new_h) # generate a random top coordinate
    right = left + new_w
    bottom = top + new_h # generate a random bottom coordinate

    cropped = img.crop((left, top, right, bottom)) # crop the image
    return cropped.resize((w, h), Image.BILINEAR)


def jpeg_recompress(img, quality_min=40, quality_max=70):
    # re-encode the image as JPEG with lower quality to simulate repost compression
    quality = random.randint(quality_min, quality_max) 
    buffer = io.BytesIO() 
    img.convert("RGB").save(buffer, format="JPEG", quality=quality) 
    buffer.seek(0) 
    recompressed = Image.open(buffer) 
    return recompressed.convert("RGB") # convert the image to RGB


def color_adjust(img, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1)):
    # adjust the color of the image
    b_factor = random.uniform(*brightness_range) 
    c_factor = random.uniform(*contrast_range) 
    img = ImageEnhance.Brightness(img).enhance(b_factor) 
    img = ImageEnhance.Contrast(img).enhance(c_factor) 
    return img # return the adjusted image


def blur(img, radius_min=0.5, radius_max=1.5):
    # blur the image
    radius = random.uniform(radius_min, radius_max) # generate a random radius between the minimum and maximum radius
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_border(img, border_size_ratio=0.03, color="white"):
    # add a border to the image      
    w, h = img.size 
    border = int(min(w, h) * border_size_ratio) 
    if border <= 0: # if the border is less than or equal to 0, return the image
        return img
    return ImageOps.expand(img, border=border, fill=color) # add the border to the image


def add_emoji_overlay(img, size_ratio=0.18):
    # add an emoji overlay to the image
    w, h = img.size  
    emoji_size = int(min(w, h) * size_ratio) 
    if emoji_size <= 0:
        return img

    img = img.copy()
    draw = ImageDraw.Draw(img) # create a draw object

    margin = int(emoji_size * 0.15) # generate a random margin based on the emoji size
    x1 = w - emoji_size - margin 
    y1 = h - emoji_size - margin 
    x2 = x1 + emoji_size 
    y2 = y1 + emoji_size 

    # Face circle in the bottom-right corner of the image
    draw.ellipse((x1, y1, x2, y2), fill=(255, 221, 87), outline=(0, 0, 0))

    # Eyes in the bottom-right corner of the image
    eye_radius = emoji_size * 0.07
    eye_offset_x = emoji_size * 0.22
    eye_offset_y = emoji_size * 0.25

    ex1 = x1 + eye_offset_x
    ey1 = y1 + eye_offset_y
    ex2 = ex1 + eye_radius
    ey2 = ey1 + eye_radius

    ex1b = x2 - eye_offset_x - eye_radius
    ey1b = ey1
    ex2b = ex1b + eye_radius
    ey2b = ey1b + eye_radius

    draw.ellipse((ex1, ey1, ex2, ey2), fill=(0, 0, 0))
    draw.ellipse((ex1b, ey1b, ex2b, ey2b), fill=(0, 0, 0))

    # Neutral mouth
    mouth_width = emoji_size * 0.45
    mouth_height = emoji_size * 0.06
    mx1 = (x1 + x2) / 2 - mouth_width / 2
    my1 = y1 + emoji_size * 0.60
    mx2 = mx1 + mouth_width
    my2 = my1 + mouth_height
    draw.rectangle((mx1, my1, mx2, my2), fill=(0, 0, 0))

    return img



IN_DISTRIBUTION_EDITS = [
    "minor_crop",
    "jpeg_recompress",
    "color_adjust",
    "blur",
]

ALL_EDITS = [
    "minor_crop",
    "jpeg_recompress",
    "color_adjust",
    "blur",
    "add_border",
    "add_emoji_overlay",
]


def apply_edit_by_name(img, edit_name):
    # apply the edit to the image based on the edit name
    if edit_name == "minor_crop":
        return minor_crop(img)
    elif edit_name == "jpeg_recompress":
        return jpeg_recompress(img)
    elif edit_name == "color_adjust":
        return color_adjust(img)
    elif edit_name == "blur":
        return blur(img)
    elif edit_name == "add_border":
        return add_border(img)
    elif edit_name == "add_emoji_overlay":
        return add_emoji_overlay(img)
    else:
        raise ValueError(f"Unknown edit: {edit_name}")


def random_train_edit(img):
    # randomly select an edit from the in-distribution edits
    edit_name = random.choice(IN_DISTRIBUTION_EDITS)
    return apply_edit_by_name(img, edit_name)


def random_any_edit(img):
    # randomly select an edit from all edits
    edit_name = random.choice(ALL_EDITS)
    return apply_edit_by_name(img, edit_name)
