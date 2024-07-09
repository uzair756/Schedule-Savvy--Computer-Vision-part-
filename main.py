from flask import Flask, request, jsonify
import cv2
import os
import pytesseract
import re
import logging
import base64
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

cred = credentials.Certificate(r"C:\Users\raufu\Downloads\alarmls-firebase-adminsdk-4oudq-b948799ab3.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://alarmls-default-rtdb.firebaseio.com/'
})

def extract_rows(image_path, output_folder):
    try:
        image = cv2.imread(image_path)
        if image is None:
            app.logger.error("Error: Unable to load image.")
            return

        day_segments = {
            'Monday': (122, 217, 0, 1000),
            'Tuesday': (215, 310, 0, 1000),
            'Wednesday': (310, 402, 0, 1000),
            'Thursday': (400, 495, 0, 1200),
            'Friday': (495, 587, 0, 1000),
        }

        rows_folder = os.path.join(output_folder, 'rows')
        if not os.path.exists(rows_folder):
            os.makedirs(rows_folder)

        for day, (start_y, end_y, start_x, end_x) in day_segments.items():
            day_image = image[start_y:end_y, start_x:end_x]
            cv2.imwrite(os.path.join(rows_folder, f'{day}.png'), day_image)
    except Exception as e:
        app.logger.error(f"Error in extract_rows: {e}")

def segment_images(input_folder, output_folder):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith('.png'):
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)
                if image is None:
                    app.logger.error(f"Error: Unable to load image {filename}.")
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

                lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    app.logger.debug("No lines detected using HoughLinesP.")

                cv2.imwrite(os.path.join(output_folder, 'debug_lines.png'), image)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rectangles = [cv2.boundingRect(contour) for contour in contours]
                rectangles = sorted(rectangles, key=lambda x: x[0])

                for i, (x, y, w, h) in enumerate(rectangles):
                    segmented_image = image[y:y + h, x:x + w]
                    day = os.path.splitext(filename)[0]
                    day_folder = os.path.join(output_folder, day)
                    if not os.path.exists(day_folder):
                        os.makedirs(day_folder)
                    segment_path = os.path.join(day_folder, f'segment_{i}.jpg')
                    cv2.imwrite(segment_path, segmented_image)
                    app.logger.debug(f"Saved segmented image: {segment_path}")
    except Exception as e:
        app.logger.error(f"Error in segment_images: {e}")

def extract_text(input_folder):
    try:
        if not os.path.exists('text_files'):
            os.makedirs('text_files')

        for day_folder in os.listdir(input_folder):
            if os.path.isdir(os.path.join(input_folder, day_folder)):
                day_text = ''
                for filename in os.listdir(os.path.join(input_folder, day_folder)):
                    if filename.endswith('.jpg'):
                        image_path = os.path.join(input_folder, day_folder, filename)
                        image = cv2.imread(image_path)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                        preprocessed_image_path = 'preprocessed_image.png'
                        cv2.imwrite(preprocessed_image_path, binary_image)
                        text = pytesseract.image_to_string(binary_image, config='--psm 6')
                        day_text += text + '\n'

                with open(os.path.join('text_files', f'{day_folder}.txt'), 'w') as f:
                    f.write(day_text)
                app.logger.debug(f"Text extracted and saved to {os.path.join('text_files', f'{day_folder}.txt')}")
    except Exception as e:
        app.logger.error(f"Error in extract_text: {e}")

def adjust_time(time_str):
    hour, minute = map(int, time_str.split(":"))
    hour -= 1
    if hour < 0:
        hour = 23
    return f"{hour:02}:{minute:02} AM"

def extract_information(input_folder, output_folder):
    try:
        subjects_dict = {
            "computer vision": "computer vision",
            "computer networks": "computer networks",
            "technical writing": "technical writing",
            "artificial intelligence": "artificial intelligence",
            "mobile application development": "mobile application development",
            "technical": "technical",
            "artificial intelligence lab": "artificial intelligence lab",
        }

        time_pattern = re.compile(r'\b\d{4}-\d{4}\b')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_folder, filename)
                day = filename.split('.')[0]

                timetable_info = ''
                current_subject_lines = []
                subject_found = False
                current_subject = None

                output_file_path = os.path.join(output_folder, f'{day}.txt')

                # Read existing entries to avoid duplicates
                existing_entries = set()
                if os.path.exists(output_file_path):
                    with open(output_file_path, 'r') as output_file:
                        lines = output_file.readlines()
                        for i in range(0, len(lines), 2):
                            subject = lines[i].strip()
                            time = lines[i + 1].strip()
                            existing_entries.add((subject, time))

                with open(file_path, 'r') as file:
                    lines = file.readlines()

                    for line in lines:
                        line = line.strip().lower()  # Convert line to lowercase

                        # Collect potential subject lines
                        current_subject_lines.append(line)
                        if len(current_subject_lines) > 3:  # Keep only last 3 lines
                            current_subject_lines.pop(0)

                        # Check if the collected lines contain a subject
                        combined_subject = ' '.join(current_subject_lines).replace('\n', ' ').strip()
                        for subject in subjects_dict.keys():
                            if subject in combined_subject:
                                current_subject = subjects_dict[subject]
                                current_subject_lines = []  # Reset for next subject
                                subject_found = True
                                break

                        # Check if the line matches the time pattern and a subject was found
                        if subject_found and time_pattern.search(line):
                            time = time_pattern.search(line).group()
                            start_time = adjust_time(f"{time[:2]}:{time[2:4]}")
                            end_time = adjust_time(f"{time[5:7]}:{time[7:9]}")
                            entry = (f"subject: {current_subject}\n", f"time: {start_time}\n")
                            if entry not in existing_entries:
                                timetable_info += entry[0] + entry[1]
                                existing_entries.add(entry)
                            current_subject = None
                            subject_found = False  # Reset to find next subject

                # Write the extracted information to a file for each day
                with open(output_file_path, 'w') as output_file:
                    output_file.write(timetable_info)

    except Exception as e:
        logging.error(f"Error in extract_information: {e}")

def upload_to_firebase():
    try:
        timetable_folder = 'timetable'
        for filename in os.listdir(timetable_folder):
            if filename.endswith('.txt'):
                day = filename.split('.')[0].capitalize()  # Ensure day is capitalized correctly
                file_path = os.path.join(timetable_folder, filename)

                with open(file_path, 'r') as file:
                    lines = file.readlines()

                for i in range(0, len(lines), 2):  # Each entry spans 2 lines (subject, time)
                    try:
                        day_info = {
                            'day': day.capitalize(),  # Capitalize the first letter of the day
                            'subject': lines[i].replace("subject:", "").strip(),
                            'time': lines[i + 1].replace("time:", "").strip()
                        }
                    except IndexError:
                        continue

                    app.logger.debug(f"Uploading {day_info}")  # Debug output before uploading
                    timetable_ref = db.reference('timetable')
                    new_entry_ref = timetable_ref.push()  # Generate a new unique key
                    new_entry_ref.set(day_info)
                    app.logger.debug(f"Timetable for {day} uploaded to Firebase Realtime Database successfully.")

    except Exception as e:
        app.logger.error(f"Error in upload_to_firebase: {e}")

@app.route('/extract_timetable', methods=['POST'])
def extract_timetable():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image file provided'}), 400

        encoded_image = data['image']
        image_bytes = base64.b64decode(encoded_image)
        image_path = 'uploaded_image.jpg'
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        app.logger.debug(f"Image saved to {image_path}")

        output_folder = 'output'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        extract_rows(image_path, output_folder)
        segment_images(os.path.join(output_folder, 'rows'), output_folder)
        extract_text(output_folder)
        extract_information('text_files', 'timetable')
        upload_to_firebase()

        return jsonify({'message': 'Timetable extraction and upload completed successfully.'})
    except Exception as e:
        app.logger.error(f"Error in extract_timetable: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)