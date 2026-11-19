# Smart Energy Monitor

A smart energy monitoring project with a Python application and ESP firmware.

## Project files

- `main.py` — Python application
- `Esp_code/Esp_code.ino` — ESP firmware

## ESP setup

Copy `Esp_code/secrets.example.h` to `Esp_code/secrets.h` and set your Wi-Fi name and password in the local copy. The real `secrets.h` file is ignored by Git and should never be committed.

Update `SERVER_URL` in the firmware to the address of the machine running the Python application.

Local databases, Python environments, caches, environment files, and ESP Wi-Fi credentials are excluded from version control.
