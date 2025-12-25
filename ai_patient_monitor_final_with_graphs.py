# ai_patient_monitor_final_with_graphs.py
# Final script (live monitoring + metrics + CSV & Excel export + summary bar charts)
# - Two windows (one per patient), 5 stacked subplots each
# - Metrics recorded for all 5 vitals and printed once at the end (wide table)
# - Dynamic precautions + overlay annotations
# - SMS enabled with cooldown, beep on SMS send (one beep per SMS)
# - Temperature included in metrics and anomaly detection
# - At end: save metrics to CSV & Excel and show summary bar charts per patient

import pandas as pd
import zipfile
import random
import time
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import datetime
import os

# optional beep for Windows
try:
    import winsound
    _HAS_WINSOUND = True
except Exception:
    _HAS_WINSOUND = False

from twilio.rest import Client

# ---------------------------- USER CONFIG ----------------------------
ZIP_PATH = r"C:\Users\DELL\Downloads\human_vital_signs_dataset_2024.csv.zip"

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
DOCTOR_NUMBER = os.getenv("DOCTOR_NUMBER")


MONITOR_DURATION_SECONDS = 120    # 2 minutes
UPDATE_INTERVAL_SECONDS = 3       # sleep between updates
SMS_COOLDOWN_SECONDS = 60         # cooldown per patient+vital
N_PATIENTS = 2
MAX_POINTS = 60                   # number of history points to keep in plots
ANNOTATION_TTL = 8                # seconds to keep overlay precaution on plot

# Save outputs
OUTPUT_DIR = os.path.join(os.getcwd(), "monitor_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Beep behavior: beep when SMS is successfully sent
BEEP_ON = True

# ---------------------------- TWILIO SETUP ----------------------------
client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_sms_alert(patient_id, critical_vitals):
    """
    Send SMS through Twilio. Returns True if send attempted (no exception),
    otherwise False.
    """
    try:
        body = f"ALERT: Patient {patient_id} critical: {', '.join(critical_vitals)}"
        message = client.messages.create(body=body, from_=TWILIO_NUMBER, to=DOCTOR_NUMBER)
        print(f"[{datetime.datetime.now()}] SMS sent for Patient {patient_id} (SID: {message.sid})")
        return True
    except Exception as e:
        print(f"[{datetime.datetime.now()}] SMS failed: {e}")
        return False

# ---------------------------- VITAL THRESHOLDS & ORDER ----------------------------
VITAL_THRESHOLDS = {
    'Heart Rate': {'warning': (55, 100), 'critical': (45, 115)},
    'Body Temperature': {'warning': (36.0, 37.5), 'critical': (35.0, 38.5)},
    'Oxygen Saturation': {'warning': (94, 100), 'critical': (90, 100)},
    'Systolic Blood Pressure': {'warning': (90, 130), 'critical': (80, 160)},
    'Diastolic Blood Pressure': {'warning': (60, 85), 'critical': (50, 100)}
}

# exact order requested for metrics and plotting
vital_list = [
    'Heart Rate',
    'Oxygen Saturation',
    'Body Temperature',
    'Systolic Blood Pressure',
    'Diastolic Blood Pressure'
]

# ---------------------------- COLORS (fixed) ----------------------------
COLOR_MAP = {
    'Heart Rate': 'red',
    'Oxygen Saturation': 'blue',
    'Body Temperature': 'orange',
    'Systolic Blood Pressure': 'green',
    'Diastolic Blood Pressure': 'purple'
}

# ---------------------------- PRECAUTIONS (base text, ASCII-safe) ----------------------------
PRECAUTIONS = {
    'Heart Rate': [
        "Make the patient lie down and rest.",
        "Check pulse manually and record rate.",
        "Loosen tight clothing; keep patient calm.",
        "If irregular/very high/very low persists, arrange emergency medical support."
    ],
    'Oxygen Saturation': [
        "Sit the patient upright to help breathing.",
        "Ensure fresh air and loosen tight clothing around chest.",
        "Monitor breathing rate; provide supplemental oxygen if available.",
        "If no improvement, call emergency services."
    ],
    'Body Temperature': [
        "Remove excess clothing/blankets if hot, or provide warm coverings if cold.",
        "Measure temperature manually and repeat after 10-15 minutes.",
        "Hydrate patient if feverish and seek medical advice if very high or low."
    ],
    'Systolic Blood Pressure': [
        "Have the patient rest and avoid standing suddenly.",
        "Record BP again after a few minutes.",
        "If very high or symptomatic, seek urgent medical attention."
    ],
    'Diastolic Blood Pressure': [
        "Ensure patient is resting and calm.",
        "Avoid strenuous activity until BP is stable.",
        "Seek medical advice if diastolic is very high/low or symptoms occur."
    ]
}

# ---------------------------- DATA LOADING ----------------------------
print("Loading dataset from zip...")
with zipfile.ZipFile(ZIP_PATH) as z:
    csv_filename = z.namelist()[0]
    with z.open(csv_filename) as f:
        data = pd.read_csv(f)

print("Dataset loaded. Preparing patients...\n")

# Pick first N patients
patients = data.head(N_PATIENTS).copy().reset_index(drop=True)

required_cols = ['Patient ID'] + list(VITAL_THRESHOLDS.keys())
for c in required_cols:
    if c not in patients.columns:
        raise KeyError(f"Required column '{c}' not found in dataset columns: {patients.columns.tolist()}")

# Ensure Patient ID remains integer (for display) and convert vital columns to float to avoid dtype warnings
patients['Patient ID'] = patients['Patient ID'].astype(int)
for v in vital_list:
    patients[v] = pd.to_numeric(patients[v], errors='coerce').astype(float)

# ---------------------------- PLOT SETUP: two separate figures ----------------------------
plt.ion()
patient_figs = []
patient_axes = []  # list of axes arrays for each patient
lines = {}         # (patient_idx, vital) -> Line2D
data_queues = {}   # (patient_idx, vital) -> deque
time_queues = [deque(maxlen=MAX_POINTS) for _ in range(N_PATIENTS)]

for p_idx in range(N_PATIENTS):
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(9, 11), sharex=True)
    fig.suptitle(f"Patient {patients.loc[p_idx, 'Patient ID']} — Live Vitals")
    patient_figs.append(fig)
    patient_axes.append(axes)

    for ax, vital in zip(axes, vital_list):
        ax.set_ylabel(vital)
        ax.grid(True)
        label = f"P{patients.loc[p_idx,'Patient ID']}"
        dq = deque(maxlen=MAX_POINTS)
        init_val = patients.loc[p_idx, vital]
        if pd.isna(init_val):
            init_val = 0.0
        dq.extend([float(init_val)] * 3)
        data_queues[(p_idx, vital)] = dq
        line, = ax.plot(list(range(len(dq))), list(dq), label=label, color=COLOR_MAP[vital])
        lines[(p_idx, vital)] = line
        ax.legend(loc='upper right', fontsize='small')

    axes[-1].set_xlabel("Time (points)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.pause(0.05)

# ---------------------------- ANOMALY / SMS MANAGEMENT ----------------------------
# TRIGGER vitals (which cause alerts/SMS/beep) — per last instruction include all five
TRIGGER_VITALS = set(vital_list)  # includes temperature as requested

last_sms_time = defaultdict(lambda: datetime.datetime.min)

def is_critical(vital, value):
    low_crit, high_crit = VITAL_THRESHOLDS[vital]['critical']
    return (value < low_crit) or (value > high_crit)

# Dynamic severity & precaution generator
def dynamic_precaution(vital, value):
    v = float(value)
    if vital == 'Heart Rate':
        if v < 40:
            severity = "Severe Low"
            advice = [
                "Severe bradycardia - call emergency.",
                "Check airway, breathing & pulse immediately.",
                "Place patient supine and elevate legs if dizzy."
            ]
        elif 40 <= v < 55:
            severity = "Low"
            advice = [
                "Bradycardia - have patient rest & monitor.",
                "Check medication history and measure pulse manually.",
                "If symptomatic or persists, seek urgent care."
            ]
        elif 55 <= v <= 100:
            severity = "Normal"
            advice = ["Heart rate within normal limits. Continue monitoring."]
        elif 100 < v <= 140:
            severity = "High"
            advice = [
                "Tachycardia - make patient sit/lie down and rest.",
                "Check for fever, pain or anxiety as causes.",
                "If sustained, seek medical assessment."
            ]
        else:
            severity = "Severe High"
            advice = [
                "Severe tachycardia - call emergency services.",
                "Keep patient calm, check airway/breathing.",
                "Prepare for urgent transfer if condition doesn't improve."
            ]
    elif vital == 'Oxygen Saturation':
        if v < 85:
            severity = "Severe Low"
            advice = [
                "Severely low SpO2 - provide high-flow oxygen if available.",
                "Sit patient upright to aid breathing and call emergency.",
                "Monitor respiratory rate closely."
            ]
        elif 85 <= v < 90:
            severity = "Low"
            advice = [
                "Low SpO2 - provide supplemental oxygen if available.",
                "Sit patient upright and ensure airway is clear.",
                "Avoid exertion; seek urgent medical help if no improvement."
            ]
        elif 90 <= v < 94:
            severity = "Borderline"
            advice = [
                "Borderline oxygen - monitor closely.",
                "Encourage slow deep breaths; ensure good ventilation.",
                "If drops further, escalate care."
            ]
        else:
            severity = "Normal"
            advice = ["Oxygen saturation within normal limits. Continue monitoring."]
    elif vital == 'Body Temperature':
        if v < 35.0:
            severity = "Severe Low"
            advice = [
                "Severely low temperature - warm patient and seek urgent care.",
                "Cover with blankets and check for signs of hypothermia."
            ]
        elif 35.0 <= v < 36.0:
            severity = "Low"
            advice = [
                "Low temperature - provide warmth and re-check.",
                "Monitor and seek advice if remains low."
            ]
        elif 36.0 <= v <= 37.5:
            severity = "Normal"
            advice = ["Body temperature within normal range. Continue monitoring."]
        elif 37.5 < v <= 38.5:
            severity = "Fever"
            advice = [
                "Mild fever - hydrate patient and re-check temperature.",
                "Seek medical advice if fever persists or very high."
            ]
        else:
            severity = "High Fever"
            advice = [
                "High fever - seek urgent medical attention if very high.",
                "Consider antipyretic if advised and monitor closely."
            ]
    elif vital == 'Systolic Blood Pressure':
        if v < 80:
            severity = "Severe Low"
            advice = [
                "Severely low systolic BP - ensure patient is supine.",
                "Check for bleeding or shock; call emergency services."
            ]
        elif 80 <= v < 90:
            severity = "Low"
            advice = [
                "Low systolic BP - rest patient, re-measure after a few minutes.",
                "Check for dizziness; seek medical advice if symptomatic."
            ]
        elif 90 <= v <= 130:
            severity = "Normal"
            advice = ["Systolic BP within normal limits. Continue monitoring."]
        elif 130 < v <= 160:
            severity = "High"
            advice = [
                "Elevated systolic BP - rest and re-measure.",
                "Avoid stimulants; seek medical review if symptomatic."
            ]
        else:
            severity = "Severe High"
            advice = [
                "Severely high systolic BP - seek urgent medical attention."
            ]
    elif vital == 'Diastolic Blood Pressure':
        if v < 50:
            severity = "Severe Low"
            advice = [
                "Severely low diastolic BP - ensure patient is supine and monitor consciousness.",
                "Call emergency services if patient is unstable."
            ]
        elif 50 <= v < 60:
            severity = "Low"
            advice = [
                "Low diastolic BP - rest patient and re-measure.",
                "Monitor symptoms and seek advice if dizziness occurs."
            ]
        elif 60 <= v <= 85:
            severity = "Normal"
            advice = ["Diastolic BP within normal limits. Continue monitoring."]
        elif 85 < v <= 100:
            severity = "High"
            advice = [
                "Elevated diastolic BP - re-measure after rest and avoid activity."
            ]
        else:
            severity = "Severe High"
            advice = [
                "Severely high diastolic BP - seek urgent medical attention."
            ]
    else:
        severity = "Unknown"
        advice = ["No specific advice available."]
    base = PRECAUTIONS.get(vital, [])
    return severity, advice + base

# tracking overlay annotations per patient+vital -> (text_obj, expiry_time)
overlay_annotations = { (p, vital): None for p in range(N_PATIENTS) for vital in vital_list }

# ---------------------------- METRICS DATA STRUCTURES ----------------------------
# For each patient_idx and vital, track: samples, sum, min, max, anomaly_count
metrics = {
    p_idx: {
        vital: {
            'samples': 0,
            'sum': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'anomaly_count': 0
        } for vital in vital_list
    } for p_idx in range(N_PATIENTS)
}

# also keep a simple event log (list of dicts) for optional CSV export
event_log = []

# ---------------------------- SIMULATION LOOP ----------------------------
print("Starting 2-minute live monitoring simulation (two windows) with metrics...\n")
start_time = time.time()

while (time.time() - start_time) < MONITOR_DURATION_SECONDS:
    elapsed = int(time.time() - start_time)
    for p_idx in range(N_PATIENTS):
        time_queues[p_idx].append(elapsed)
        for vital in vital_list:
            current_val = patients.at[p_idx, vital]
            if pd.isna(current_val):
                current_val = 0.0
            fluctuation = random.uniform(-1.5, 1.5)
            new_val = max(0.0, float(current_val) + fluctuation)

            # 20% chance to inject a critical event (simulate)
            if random.random() < 0.2:
                if vital == 'Heart Rate':
                    new_val = random.choice([random.uniform(30, 45), random.uniform(120, 160)])
                elif vital == 'Body Temperature':
                    new_val = random.choice([random.uniform(34, 35), random.uniform(39, 41)])
                elif vital == 'Oxygen Saturation':
                    new_val = random.uniform(78, 89)
                elif vital == 'Systolic Blood Pressure':
                    new_val = random.choice([random.uniform(70, 85), random.uniform(160, 190)])
                elif vital == 'Diastolic Blood Pressure':
                    new_val = random.choice([random.uniform(40, 55), random.uniform(100, 120)])

            # write back (float) to patients dataframe to avoid dtype warnings
            patients.at[p_idx, vital] = float(new_val)

            # update queue + line
            dq = data_queues[(p_idx, vital)]
            dq.append(float(new_val))
            line = lines[(p_idx, vital)]
            line.set_xdata(list(range(len(dq))))
            line.set_ydata(list(dq))

            # update metrics
            m = metrics[p_idx][vital]
            m['samples'] += 1
            m['sum'] += float(new_val)
            if new_val < m['min']:
                m['min'] = float(new_val)
            if new_val > m['max']:
                m['max'] = float(new_val)

            # adjust y-limits for that axis
            fig_axes = patient_axes[p_idx]
            ax_index = vital_list.index(vital)
            ax = fig_axes[ax_index]
            if dq:
                ymin, ymax = min(dq), max(dq)
                if ymin == ymax:
                    ax.set_ylim(ymin - 1, ymax + 1)
                else:
                    pad = (ymax - ymin) * 0.2
                    ax.set_ylim(ymin - pad, ymax + pad)

            # ANOMALY DETECTION for all TRIGGER_VITALS
            if is_critical(vital, new_val):
                # increment anomaly_count for metrics
                metrics[p_idx][vital]['anomaly_count'] += 1

                patient_id = int(patients.at[p_idx, 'Patient ID'])
                severity, advice_lines = dynamic_precaution(vital, new_val)
                timestamp = datetime.datetime.now()
                display_vital = 'SpO2' if vital == 'Oxygen Saturation' else vital

                # ASCII-safe print
                print(f"[{timestamp}] Patient {patient_id} - CRITICAL ({display_vital}) = {new_val:.2f} [{severity}]")
                print("Dynamic Precautions (in absence of doctor):")
                for l in advice_lines:
                    safe_line = l.encode('ascii', errors='replace').decode('ascii')
                    print(f"- {safe_line}")
                print("-" * 60)

                # add to event log
                event_log.append({
                    "timestamp": timestamp.isoformat(),
                    "patient_id": patient_id,
                    "vital": display_vital,
                    "value": round(new_val, 2),
                    "severity": severity
                })

                # overlay precaution text on HR subplot (index 0) of this patient's figure
                overlay_ax = patient_axes[p_idx][0]
                prev = overlay_annotations.get((p_idx, vital))
                if prev is not None:
                    text_obj, expiry = prev
                    try:
                        text_obj.remove()
                    except Exception:
                        pass
                short_vital = display_vital
                short = f"{short_vital} {new_val:.1f} ({severity})\n" + "\n".join([s for s in advice_lines[:3]])
                short_ascii = short.encode('ascii', errors='replace').decode('ascii')
                text_obj = overlay_ax.text(
                    0.98, 0.95, short_ascii,
                    transform=overlay_ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(facecolor='yellow', alpha=0.85, edgecolor='black')
                )
                overlay_annotations[(p_idx, vital)] = (text_obj, time.time() + ANNOTATION_TTL)

                # SMS cooldown & send (beep only when SMS sent)
                key = (patient_id, vital)
                now = datetime.datetime.now()
                if (now - last_sms_time[key]).total_seconds() >= SMS_COOLDOWN_SECONDS:
                    critical_vitals = [f"{display_vital}: {new_val:.1f}"]
                    sent_ok = send_sms_alert(patient_id, critical_vitals)
                    if sent_ok and BEEP_ON and _HAS_WINSOUND:
                        try:
                            winsound.Beep(1000, 700)   # frequency=1000Hz, duration=700ms


                        except RuntimeError:
                            print("Beep failed(possibly muted system)")
                    last_sms_time[key] = now
                else:
                    remaining = SMS_COOLDOWN_SECONDS - (now - last_sms_time[key]).total_seconds()
                    print(f"(SMS cooldown: will allow SMS for {display_vital} of patient {patient_id} in {int(remaining)}s)")

    # Clean up expired overlay annotations
    for key, val in list(overlay_annotations.items()):
        if val is None:
            continue
        text_obj, expiry = val
        if time.time() > expiry:
            try:
                text_obj.remove()
            except Exception:
                pass
            overlay_annotations[key] = None

    # redraw both figures
    for p_idx in range(N_PATIENTS):
        fig = patient_figs[p_idx]
        axes = patient_axes[p_idx]
        # update x-limits to show last up to MAX_POINTS points
        current_max_len = max(len(data_queues[(p_idx, vital)]) for vital in vital_list)
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(0, max(10, current_max_len))
        fig.canvas.draw()
        fig.canvas.flush_events()

    time.sleep(UPDATE_INTERVAL_SECONDS)

print("\nMonitoring simulation completed (120 seconds).")

# ---------------------------- METRICS TABLE CONSTRUCTION (Option A: wide table) ----------------------------
rows = []
index = []

for p_idx in range(N_PATIENTS):
    pid_label = f"P{patients.loc[p_idx, 'Patient ID']}"
    # counts (anomaly_count)
    cnts = {v: metrics[p_idx][v]['anomaly_count'] for v in vital_list}
    rows.append(cnts)
    index.append(f"{pid_label}_anomaly_count")
    # averages
    avgs = {}
    for v in vital_list:
        m = metrics[p_idx][v]
        avg = (m['sum'] / m['samples']) if m['samples'] > 0 else float('nan')
        avgs[v] = round(avg, 2)
    rows.append(avgs)
    index.append(f"{pid_label}_avg")
    # min
    mins = {}
    for v in vital_list:
        m = metrics[p_idx][v]
        mins[v] = (round(m['min'], 2) if m['samples'] > 0 and m['min'] != float('inf') else float('nan'))
    rows.append(mins)
    index.append(f"{pid_label}_min")
    # max
    maxs = {}
    for v in vital_list:
        m = metrics[p_idx][v]
        maxs[v] = (round(m['max'], 2) if m['samples'] > 0 and m['max'] != float('-inf') else float('nan'))
    rows.append(maxs)
    index.append(f"{pid_label}_max")
    # % critical
    pcts = {}
    for v in vital_list:
        m = metrics[p_idx][v]
        pct = (m['anomaly_count'] / m['samples'] * 100) if m['samples'] > 0 else float('nan')
        pcts[v] = round(pct, 2)
    rows.append(pcts)
    index.append(f"{pid_label}_pct_critical")

metrics_df = pd.DataFrame(rows, index=index, columns=vital_list)

# Pretty print to terminal (T2 style, wide table - Option A)
print("\n\n=== METRICS SUMMARY (per patient, per vital) ===\n")
print(metrics_df)

# ---------------------------- SAVE METRICS CSV & EXCEL ----------------------------
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(OUTPUT_DIR, f"metrics_summary_{ts}.csv")
excel_path = os.path.join(OUTPUT_DIR, f"metrics_summary_{ts}.xlsx")

try:
    metrics_df.to_csv(csv_path)
    # Excel (requires openpyxl)
    metrics_df.to_excel(excel_path)
    print(f"\nMetrics saved to:\n - {csv_path}\n - {excel_path}")
except Exception as e:
    print(f"Failed to save metrics files: {e}")

# ---------------------------- SAVE EVENT LOG (optional) ----------------------------
if event_log:
    ev_df = pd.DataFrame(event_log)
    ev_csv = os.path.join(OUTPUT_DIR, f"event_log_{ts}.csv")
    try:
        ev_df.to_csv(ev_csv, index=False)
        print(f"Event log saved to: {ev_csv}")
    except Exception as e:
        print(f"Failed to save event log: {e}")
else:
    print("No events recorded to save.")

# ---------------------------- SUMMARY BAR CHARTS ----------------------------
# Two plots per patient: averages and %critical (side-by-side)
for p_idx in range(N_PATIENTS):
    pid_label = f"P{patients.loc[p_idx, 'Patient ID']}"
    # averages
    avg_row = metrics_df.loc[f"{pid_label}_avg"]
    pct_row = metrics_df.loc[f"{pid_label}_pct_critical"]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig.suptitle(f"Summary for {pid_label}")

    # bar chart: averages
    axs[0].bar(vital_list, [avg_row[v] for v in vital_list])
    axs[0].set_title("Average values (2-min)")
    axs[0].set_ylabel("Value")
    axs[0].tick_params(axis='x', rotation=20)

    # bar chart: percent critical
    axs[1].bar(vital_list, [pct_row[v] for v in vital_list])
    axs[1].set_title("% Critical (percentage of samples)")
    axs[1].set_ylabel("% Critical")
    axs[1].tick_params(axis='x', rotation=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # save summary figure
    fig_path = os.path.join(OUTPUT_DIR, f"summary_{pid_label}_{ts}.png")
    try:
        fig.savefig(fig_path)
        print(f"Saved summary chart for {pid_label} to {fig_path}")
    except Exception:
        pass

plt.show(block=True)

print("\nAll done. Outputs (metrics & charts) are in:", OUTPUT_DIR)
