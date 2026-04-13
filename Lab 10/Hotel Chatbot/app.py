from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# -----------------------------
# MEMORY (booking context)
# -----------------------------
user_context = {
    "room": None,
    "date": None,
    "booking": False
}

# =============================
# MAIN CONTROLLER FUNCTION
# =============================
def get_bot_response(user_input):
    user_input = user_input.lower().strip()

    # 1. Greeting
    response = handle_greetings(user_input)
    if response:
        return response

    # 2. Amenities ⭐ (MOVE UP FIX)
    response = handle_amenities(user_input)
    if response:
        return response

    # 3. Restaurant
    response = handle_restaurant(user_input)
    if response:
        return response

    # 4. Location
    response = handle_location(user_input)
    if response:
        return response

    # 5. ROOMS FIRST (FIX)
    response = handle_rooms(user_input)
    if response:
        return response

    # 6. BOOKING AFTER ROOMS
    response = handle_booking(user_input)
    if response:
        return response

    # 7. Prices
    response = handle_prices(user_input)
    if response:
        return response

    # 8. Availability
    response = handle_availability(user_input)
    if response:
        return response


    # 9. Fallback
    return fallback_response()


# =============================
# 1. GREETING
# =============================
def handle_greetings(text):
    if text in ["hi", "hello", "hey"]:
        return "Hello! Welcome to our Hotel Assistant 👋 How can I help you today?"
    return None


# =============================
# 2. LOCATION
# =============================
def handle_location(text):
    if "where exactly" in text or "exact" in text:
        return "We are located in DHA Phase 5, Lahore near Main Boulevard."

    if "where" in text or "location" in text:
        return "We are located in Lahore, Pakistan."

    return None


# =============================
# 3. ROOMS (REQUIRED FEATURE)
# =============================
def handle_rooms(text):

    # ONLY trigger when user asks generally about rooms
    if "room" in text and not any(word in text for word in ["price", "cost", "rate", "book", "booking"]):
        return (
            "We offer:\n"
            "- Standard\n"
            "- Deluxe\n"
            "- Suite 🏨\n\n"
            "👉 Try: 'price for deluxe room' or 'book a room'"
        )

    # room selection (during booking)
    if "standard" in text:
        user_context["room"] = "Standard"
        return "Standard Room selected ✅ Please provide booking date."

    if "deluxe" in text:
        user_context["room"] = "Deluxe"
        return "Deluxe Room selected ✅ Please provide booking date."

    if "suite" in text:
        user_context["room"] = "Suite"
        return "Suite Room selected ✅ Please provide booking date."

    return None


# =============================
# 4. PRICES (REQUIRED FEATURE)
# =============================
def handle_prices(text):
    if any(word in text for word in ["price", "cost", "rate", "how much"]):
        return (
            "Room Prices:\n"
            "Standard: $100\n"
            "Deluxe: $150\n"
            "Suite: $250 per night.\n\n"
            "👉 You can say: 'Book a deluxe room'"
        )
    return None


# =============================
# 5. BOOKING FLOW
# =============================
def handle_booking(text):

    # start booking
    if "book" in text or "booking" in text or "reserve" in text:
        user_context["booking"] = True
        return "Sure! Which room would you like? (Standard, Deluxe, Suite)"

    # date selection
    if user_context["room"] and is_date(text):
        user_context["date"] = text
        return f"{user_context['room']} room for {text}. Do you want to confirm booking?"

    # confirmation
    if text in ["yes", "confirm", "confirm booking", "yes please"]:
        if user_context["room"] and user_context["date"]:
            room = user_context["room"]
            date = user_context["date"]

            user_context["room"] = None
            user_context["date"] = None
            user_context["booking"] = False

            return f"🎉 Booking confirmed! {room} room booked for {date}."
        else:
            return "Please select room and date first."

    return None


# =============================
# 6. RESTAURANT (REQUIRED FEATURE)
# =============================
def handle_restaurant(text):
    if "chinese" in text:
        return "Chinese Menu 🍜: Chow Mein, Fried Rice, Manchurian"

    if "pakistani" in text:
        return "Pakistani Menu 🍗: Biryani, Karahi, BBQ"

    if "continental" in text:
        return "Continental Menu 🍝: Pasta, Burgers, Steak"

    if "food" in text or "restaurant" in text or "menu" in text:
        return "We offer Pakistani, Chinese, and Continental food 🍽️ Room service available 24/7."

    return None


# =============================
# 7. AMENITIES (NEW REQUIRED FEATURE ⭐)
# =============================
def handle_amenities(text):
    if "amenities" in text or "facilities" in text:
        return (
            "Hotel Amenities 🏨:\n"
            "- Free WiFi 📶\n"
            "- Swimming Pool 🏊\n"
            "- Gym 💪\n"
            "- 24/7 Room Service 🛎️\n"
            "- Free Parking 🚗"
        )
    return None


# =============================
# 8. AVAILABILITY
# =============================
def handle_availability(text):
    if any(word in text for word in ["available", "15", "20", "25", "jan", "feb"]):
        if user_context["room"]:
            return f"Yes ✅ {user_context['room']} room is available for {text}."
        else:
            return "Please select a room first."
    return None


# =============================
# 9. FALLBACK
# =============================
def fallback_response():
    return (
        "I'm not sure I understand 🤔\n"
        "You can try:\n"
        "- 'Show room prices'\n"
        "- 'Book a room'\n"
        "- 'Amenities'\n"
        "- 'Restaurant menu'"
    )

# =============================
# 10. HELPER FUNCTION TO CHECK IF TEXT CONTAINS A DATE (NEW REQUIRED FEATURE ⭐)
# =============================
def is_date(text):
    text = text.lower().strip()

    months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

    words = text.split()

    has_month = any(m in text for m in months)
    has_number = any(w.isdigit() for w in words)

    # Case 1: "15 march", "march 15", "12 feb"
    if has_month and has_number:
        return True

    # Case 2: "15th march", "15 march 2026"
    for w in words:
        if any(m in w for m in months):
            return True

    # Case 3: just number but ONLY if booking already started
    if user_context["room"] and len(words) == 1 and words[0].isdigit():
        return True

    return False

# =============================
# FLASK ROUTES
# =============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot():
    user_input = request.json.get("message")
    response = get_bot_response(user_input)
    return jsonify({"reply": response})


if __name__ == "__main__":
    app.run(debug=True)