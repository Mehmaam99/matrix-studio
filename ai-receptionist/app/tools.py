def check_appointment_availability(date: str) -> dict:
    """Check demo appointment availability."""
    print(f"TOOL EXECUTED: check_appointment_availability({date})")

    return {
        "date": date,
        "available": True,
        "message": "Demo slot available. No real booking has been made.",
    }


def get_clinic_hours() -> dict:
    """Return demo clinic opening hours."""
    print("TOOL EXECUTED: get_clinic_hours()")

    return {
        "opening_time": "09:00",
        "closing_time": "17:00",
        "days": "Monday to Saturday",
    }


TOOL_REGISTRY = {
    "check_appointment_availability": check_appointment_availability,
    "get_clinic_hours": get_clinic_hours,
}