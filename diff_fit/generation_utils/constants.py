EUROPEAN_COUNTRIES = [
    "Albania",
    "Andorra",
    "Austria",
    "Belarus",
    "Belgium",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Italy",
    "Kosovo",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Moldova",
    "Monaco",
    "Montenegro",
    "Netherlands",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Russia",
    "San Marino",
    "Serbia",
    "Slovakia",
    "Slovenia",
]

MAX_SEED = 9999999999999999


NEGATIVE_PROMPT = "octane render, render, drawing, cartoon, painting, anime, bad photo, bad photography, worst quality, low quality, blurry, bad teeth, deformed teeth, deformed lips, bad anatomy, bad proportions, deformed iris, deformed pupils, deformed eyes, bad eyes, deformed face, ugly face, bad face, morbid, mutilated, mutation, disfigured"

SERVER_PORT = 7860

GENERATION_DROPDOWN = {
    "Ethnicity": ["", "asian", "black", "white"],
    "Sex": ["", "male", "female"],
    "Hair length": ["", "bald", "short", "medium", "long"],
    "Hair color": [
        "",
        # "black",
        "light brown",
        "dark brown",
        "blonde",
        "gray",
    ],
    "Hair style": ["", "straight", "curly", "wavy"],
    "Eye color": ["", "brown", "blue", "green", "gray"],
    "Glasses": ["", "no glasses", "sunglasses", "eyeglasses"],
    "Facial hair": [
        "",
        "no facial hair",
        "beard",
        "mustache",
        "beard and mustache",
        "shaven",
    ],
}

HAIR_COLORS = {
    # "black": ["black"],
    "light brown": ["light brown", "sandy brown", "beige brown", "sunlit brown"],
    "dark brown": ["dark brown", "espresso brown", "mahogany brown", "deep chestnut"],
    "blonde": ["blonde", "champagne blonde", "butter blonde", "sandy blonde"],
    "gray": ["gray", "slate gray", "ash gray", "dove gray"],
    # "white": ["white", "snow white", "pearl white", "frost white"],
}

HAIR_LENGTHS = {
    "bald": ["bald"],
    "short": [
        "short",
        "very short",
    ],
    "medium": ["medium-length"] * 8 + ["short"] + ["long"],
    "long": ["long"] * 7 + ["very long"] * 2 + ["medium-length"],
}

HAIR_STYLES = {
    "curly": [
        "curly",
        "curly bob",
        "loose curls",
        "medium tight ringlets",
        "curly pixie cut",
        "layered curly shag",
        "curly fringe with waves",
        "spiral curls",
        "voluminous afro curls",
        "half-up curly bun",
        "curly ponytail",
        "curly updo",
        "side-parted curly",
        "curly with highlights",
        "curly mohawk",
    ],
    "straight": [
        "straight",
        "sleek ponytail",
        "blunt cut",
        "center part",
        "side part",
        "sleek bun",
        "layered straight",
        "choppy bangs",
        "wispy bangs",
        "half-up top knot",
        "braided crown",
        "slicked back style",
        "asymmetrical cut",
        "straight bob with layers",
        "tucked behind ears",
        "over the ears",
    ],
    "wavy": [
        "wavy",
        "beach waves",
        "side-swept waves",
        "soft waves with layers",
        "loose textured waves",
        "wavy with braids",
        "wavy ponytail",
        "half-up waves",
        "deep waves with side part",
        "textured bun with waves",
        "wavy curtain bangs",
        "wavy shag",
        "windswept waves",
        "glamorous hollywood waves",
        "messy wavy updo",
        "wavy with highlights",
    ],
}

EYE_COLORS = {
    "brown": [
        "hazel brown",
        "amber brown",
        "golden brown",
        "light brown",
        "chocolate brown",
        "deep brown",
        "espresso brown",
        "dark brown",
    ],
    "blue": ["sapphire blue", "sky blue", "icy blue", "blue"],
    "green": ["emerald green", "olive green", "jade green", "green"],
    "gray": ["silver gray", "steel gray", "smoky gray", "gray"],
}

GLASSES = {
    "no glasses": [""],
    "eyeglasses": [
        "round eyeglasses",
        "square eyeglasses",
        "rectangle eyeglasses",
        "oval eyeglasses",
        "eyeglasses",
    ],
    "sunglasses": [
        "sunglasses",
        "round sunglasses",
        "square sunglasses",
        "rectangle sunglasses",
        "oval sunglasses",
        "pointy sunglasses",
    ],
}

FACIAL_HAIR = {
    "no facial hair": [""],
    "beard": ["full beard", "stubble beard", "goatee beard", "beard"],
    "mustache": [
        "handlebar mustache",
        "pencil mustache",
        "walrus mustache",
        "mustache",
    ],
    "beard and mustache": [
        "goatee beard with mustache",
        "full beard with mustache",
        "trimmed beard and mustache",
        "beard and mustache",
    ],
    "shaven": ["shaven"],
}
