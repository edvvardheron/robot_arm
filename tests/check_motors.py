#!/usr/bin/env python3
"""
Feetech STS3215 servo diagnostics for SO-101 arms.

Reads status/error flags, voltage, temperature, load, current, and
position for one or more servos on the bus, and decodes what (if
anything) has tripped.

Install the SDK first (either works — same `scservo_sdk` import name):
    pip install feetech-servo-sdk
    # or the official port:
    pip install ftservo-python-sdk

Usage:
    python servo_diagnostics.py --port /dev/ttyUSB0 --ids 1 2 3 4 5 6
    # On Windows, port looks like COM4
    # On macOS, something like /dev/cu.usbmodemXXXX

If you only have one servo acting up, just pass its ID:
    python servo_diagnostics.py --port /dev/ttyUSB0 --ids 3
"""

import argparse
import sys

from scservo_sdk import sms_sts, PortHandler

# --- STS3215 control table addresses (SRAM/EEPROM per Feetech's memory map) ---
ADDR_TORQUE_ENABLE       = 40
ADDR_PRESENT_POSITION    = 56  # 2 bytes, 0-4095 = 0-360 degrees
ADDR_PRESENT_SPEED       = 58  # 2 bytes
ADDR_PRESENT_LOAD        = 60  # 2 bytes
ADDR_PRESENT_VOLTAGE     = 62  # 1 byte, units of 0.1V
ADDR_PRESENT_TEMPERATURE = 63  # 1 byte, degrees C
ADDR_SERVO_STATUS        = 65  # 1 byte, error bitmask
ADDR_MOVING              = 66  # 1 byte, 1 = moving
ADDR_PRESENT_CURRENT     = 69  # 2 bytes, units of ~6.5mA

ERROR_FLAGS = {
    0: "Voltage error — input voltage outside the configured min/max limit",
    1: "Sensor/angle error — encoder reading out of expected range",
    2: "Temperature error — over the max temperature limit",
    3: "Current error — overcurrent detected",
    5: "Overload error — sustained excess load / stalled against something",
}

DEFAULT_BAUDRATE = 1_000_000  # STS3215 factory default


def diagnose_servo(servo, servo_id):
    print(f"\n=== Servo ID {servo_id} ===")

    # Ping first — confirms it's alive on the bus at all, separate from
    # whether it will move.
    model_number, comm_result, error = servo.ping(servo_id)
    if comm_result != 0:
        print(f"  \u2717 No response to ping (comm result code: {comm_result}).")
        print("    Double-check wiring, servo ID, and baud rate before")
        print("    assuming it's actually damaged.")
        return
    print(f"  \u2713 Responded to ping (model number: {model_number})")

    # Status / error byte — this is the important one
    status, comm_result, error = servo.read1ByteTxRx(servo_id, ADDR_SERVO_STATUS)
    if comm_result != 0:
        print("  \u2717 Could not read status register.")
    elif status == 0:
        print("  \u2713 No error flags set")
    else:
        print(f"  \u26a0 Status byte: {status} (binary: {bin(status)})")
        for bit, description in ERROR_FLAGS.items():
            if status & (1 << bit):
                print(f"    \u2717 {description}")

    # Voltage
    voltage, comm_result, _ = servo.read1ByteTxRx(servo_id, ADDR_PRESENT_VOLTAGE)
    if comm_result == 0:
        print(f"  Voltage: {voltage / 10:.1f} V")

    # Temperature
    temp, comm_result, _ = servo.read1ByteTxRx(servo_id, ADDR_PRESENT_TEMPERATURE)
    if comm_result == 0:
        print(f"  Temperature: {temp} \u00b0C")

    # Present load (signed-ish duty cycle indicator)
    load, comm_result, _ = servo.read2ByteTxRx(servo_id, ADDR_PRESENT_LOAD)
    if comm_result == 0:
        print(f"  Present load: {load}")

    # Present current (~6.5mA per unit per Feetech's control table)
    current, comm_result, _ = servo.read2ByteTxRx(servo_id, ADDR_PRESENT_CURRENT)
    if comm_result == 0:
        print(f"  Present current: {current * 6.5:.0f} mA")

    # Present position
    position, comm_result, _ = servo.read2ByteTxRx(servo_id, ADDR_PRESENT_POSITION)
    if comm_result == 0:
        print(f"  Present position: {position} (0-4095 = 0-360\u00b0)")

    # Moving flag
    moving, comm_result, _ = servo.read1ByteTxRx(servo_id, ADDR_MOVING)
    if comm_result == 0:
        print(f"  Moving: {'yes' if moving else 'no'}")

    # Torque enable state
    torque, comm_result, _ = servo.read1ByteTxRx(servo_id, ADDR_TORQUE_ENABLE)
    if comm_result == 0:
        print(f"  Torque enabled: {'yes' if torque else 'no'}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Feetech STS3215 servos on an SO-101 arm."
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0 or COM4")
    parser.add_argument(
        "--ids", nargs="+", type=int, required=True,
        help="Servo IDs to check, e.g. --ids 1 2 3 4 5 6"
    )
    parser.add_argument(
        "--baud", type=int, default=DEFAULT_BAUDRATE,
        help=f"Baud rate (default {DEFAULT_BAUDRATE})"
    )
    args = parser.parse_args()

    port_handler = PortHandler(args.port)
    servo = sms_sts(port_handler)

    if not port_handler.openPort():
        print(f"Failed to open port {args.port}")
        sys.exit(1)
    if not port_handler.setBaudRate(args.baud):
        print(f"Failed to set baud rate {args.baud}")
        sys.exit(1)

    for servo_id in args.ids:
        diagnose_servo(servo, servo_id)

    port_handler.closePort()


if __name__ == "__main__":
    main()