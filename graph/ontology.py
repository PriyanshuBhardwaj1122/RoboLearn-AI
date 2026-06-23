"""
graph/ontology.py
Fixed domain ontology for RoboLearn AI knowledge graph.

Defines:
  - ENTITY_TYPES      : canonical entity type list (LLM must pick from this)
  - RELATION_TYPES    : canonical relation type list
  - TYPE_ALIASES      : exact string → canonical type mappings
  - FUZZY_RULES       : substring/keyword → canonical type mappings
  - PROTECTED_PATTERNS: regex patterns for hardware identifiers that must
                        never be merged (ADC0-5, Timer0-2, pins, etc.)
  - GENERIC_STOPWORDS : entity names that should be discarded
  - OntologyMapper    : class that normalises raw LLM output to canonical types
"""
from __future__ import annotations
import re
from typing import Optional


# ── Canonical entity types ────────────────────────────────────────────────────
# LLM is forced to pick from this list. Unknown types default to Component.

ENTITY_TYPES = {
    "Register",        # hardware registers: TCCR1A, ADMUX, DDRB
    "Pin",             # physical pins: SDA, SCL, D9, A4, MOSI
    "Protocol",        # communication protocols: I2C, SPI, UART, PWM, CAN
    "Peripheral",      # on-chip peripherals: Timer0, ADC, USART, TWI
    "Component",       # physical components: resistor, capacitor, crystal
    "Module",          # functional modules: H-bridge, LDO regulator
    "Signal",          # electrical signals: clock, interrupt, PWM output
    "Parameter",       # values/specs: 16MHz, 5V, 2A, 32KB, 10-bit
    "Interface",       # physical interfaces: USB, ICSP header, SPI bus
    "Software",        # software artifacts: sketch, library, driver, firmware
    "Package",         # ROS packages: rosserial, roscpp, rospy, std_msgs
    "Topic",           # ROS topics: cmd_vel, /odom, /scan, joint_states
    "Node",            # ROS nodes: rosserial_python, joint_state_publisher
    "Microcontroller", # MCUs: ATmega328P, ATmega168, ATmega16U2
    "Document",        # documents/manuals: datasheet, user manual, wiki
}

# Default fallback type for unresolved entities
DEFAULT_ENTITY_TYPE = "Component"


# ── Canonical relation types ──────────────────────────────────────────────────
# All extracted relation strings are normalised to this set.

RELATION_TYPES = {
    "IS_TYPE",           # entity is a type of something
    "BELONGS_TO",        # entity belongs to a parent entity
    "PART_OF",           # structural containment
    "HAS_PIN",           # component exposes a pin
    "HAS_VALUE",         # entity has a numeric/string value
    "HAS_PARAMETER",     # entity has a configuration parameter
    "CONNECTS_TO",       # physical or logical connection
    "COMMUNICATES_VIA",  # uses a protocol for communication
    "IMPLEMENTS",        # software implements a protocol/interface
    "CONTROLS",          # one entity controls another
    "READS",             # reads data from
    "WRITES",            # writes data to
    "PUBLISHES_TO",      # ROS publisher → topic
    "SUBSCRIBES_TO",     # ROS subscriber ← topic
    "REQUIRES",          # dependency
    "GENERATES",         # produces a signal/output
    "OPERATES_AT",       # frequency, voltage, speed
    "INTERFACES_WITH",   # high-level interface relationship
    "CONFIGURED_BY",     # configured via register/parameter
    "TRIGGERED_BY",      # activated by signal/interrupt
    "RELATED_TO",        # generic fallback — avoid overuse
}

# Default fallback relation type
DEFAULT_RELATION_TYPE = "RELATED_TO"


# ── Exact type alias mappings ─────────────────────────────────────────────────
# Raw LLM type string → canonical ENTITY_TYPES member
# Keys are lowercase for case-insensitive matching.

TYPE_ALIASES: dict[str, str] = {
    # Register aliases
    "register":           "Register",
    "control register":   "Register",
    "status register":    "Register",
    "data register":      "Register",
    "sfr":                "Register",
    "special function register": "Register",
    "memory mapped register": "Register",
    "io register":        "Register",
    "i/o register":       "Register",
    "configuration register": "Register",

    # Pin aliases
    "pin":                "Pin",
    "gpio":               "Pin",
    "gpio pin":           "Pin",
    "digital pin":        "Pin",
    "analog pin":         "Pin",
    "io pin":             "Pin",
    "i/o pin":            "Pin",
    "port pin":           "Pin",
    "output pin":         "Pin",
    "input pin":          "Pin",
    "pwm pin":            "Pin",

    # Protocol aliases
    "protocol":           "Protocol",
    "communication protocol": "Protocol",
    "bus protocol":       "Protocol",
    "serial protocol":    "Protocol",
    "wireless protocol":  "Protocol",
    "i2c":                "Protocol",
    "spi":                "Protocol",
    "uart":               "Protocol",
    "usart":              "Protocol",
    "can":                "Protocol",
    "pwm":                "Protocol",
    "one-wire":           "Protocol",
    "i2s":                "Protocol",

    # Peripheral aliases
    "peripheral":         "Peripheral",
    "on-chip peripheral": "Peripheral",
    "timer":              "Peripheral",
    "timer/counter":      "Peripheral",
    "counter":            "Peripheral",
    "adc":                "Peripheral",
    "analog to digital":  "Peripheral",
    "analog-to-digital":  "Peripheral",
    "dac":                "Peripheral",
    "watchdog":           "Peripheral",
    "watchdog timer":     "Peripheral",
    "wdt":                "Peripheral",
    "comparator":         "Peripheral",
    "analog comparator":  "Peripheral",
    "eeprom":             "Peripheral",
    "flash":              "Peripheral",
    "sram":               "Peripheral",

    # Microcontroller aliases
    "microcontroller":    "Microcontroller",
    "mcu":                "Microcontroller",
    "microprocessor":     "Microcontroller",
    "processor":          "Microcontroller",
    "cpu":                "Microcontroller",
    "avr":                "Microcontroller",
    "avr microcontroller": "Microcontroller",
    "chip":               "Microcontroller",
    "ic":                 "Component",
    "integrated circuit": "Component",

    # Component aliases
    "component":          "Component",
    "hardware":           "Component",
    "hardware component": "Component",
    "electronic component": "Component",
    "device":             "Component",
    "sensor":             "Component",
    "actuator":           "Component",
    "motor":              "Component",
    "servo":              "Component",
    "robot":              "Component",
    "board":              "Component",
    "shield":             "Component",
    "module":             "Module",
    "driver":             "Module",
    "motor driver":       "Module",
    "h-bridge":           "Module",
    "voltage regulator":  "Module",
    "ldo":                "Module",
    "ldo regulator":      "Module",
    "power module":       "Module",

    # Signal aliases
    "signal":             "Signal",
    "clock signal":       "Signal",
    "interrupt":          "Signal",
    "interrupt signal":   "Signal",
    "clock":              "Signal",
    "reset":              "Signal",
    "output signal":      "Signal",
    "input signal":       "Signal",
    "waveform":           "Signal",

    # Parameter aliases
    "parameter":          "Parameter",
    "specification":      "Parameter",
    "value":              "Parameter",
    "frequency":          "Parameter",
    "voltage":            "Parameter",
    "current":            "Parameter",
    "speed":              "Parameter",
    "resolution":         "Parameter",
    "baud rate":          "Parameter",
    "bit rate":           "Parameter",
    "duty cycle":         "Parameter",
    "prescaler":          "Parameter",

    # Interface aliases
    "interface":          "Interface",
    "connector":          "Interface",
    "header":             "Interface",
    "port":               "Interface",
    "bus":                "Interface",
    "usb":                "Interface",
    "usb port":           "Interface",
    "serial port":        "Interface",
    "icsp":               "Interface",
    "jtag":               "Interface",

    # Software aliases
    "software":           "Software",
    "firmware":           "Software",
    "library":            "Software",
    "sketch":             "Software",
    "code":               "Software",
    "program":            "Software",
    "script":             "Software",
    "driver software":    "Software",
    "api":                "Software",
    "framework":          "Software",
    "middleware":         "Software",
    "os":                 "Software",
    "operating system":   "Software",
    "rtos":               "Software",

    # ROS-specific aliases
    "package":            "Package",
    "ros package":        "Package",
    "catkin package":     "Package",
    "topic":              "Topic",
    "ros topic":          "Topic",
    "message":            "Topic",
    "ros message":        "Topic",
    "node":               "Node",
    "ros node":           "Node",
    "nodelet":            "Node",

    # Document aliases
    "document":           "Document",
    "datasheet":          "Document",
    "manual":             "Document",
    "user manual":        "Document",
    "reference manual":   "Document",
    "wiki":               "Document",
    "specification":      "Parameter",
    "concept":            "Component",  # remap generic "concept" to Component
}


# ── Fuzzy keyword rules ───────────────────────────────────────────────────────
# If exact alias fails, check if any keyword appears in the raw type string.
# Ordered — first match wins.

FUZZY_RULES: list[tuple[str, str]] = [
    ("register",       "Register"),
    ("pin",            "Pin"),
    ("gpio",           "Pin"),
    ("timer",          "Peripheral"),
    ("counter",        "Peripheral"),
    ("adc",            "Peripheral"),
    ("dac",            "Peripheral"),
    ("usart",          "Protocol"),
    ("uart",           "Protocol"),
    ("protocol",       "Protocol"),
    ("spi",            "Protocol"),
    ("i2c",            "Protocol"),
    ("pwm",            "Protocol"),
    ("microcontroller","Microcontroller"),
    ("processor",      "Microcontroller"),
    ("mcu",            "Microcontroller"),
    ("signal",         "Signal"),
    ("interrupt",      "Signal"),
    ("clock",          "Signal"),
    ("voltage",        "Parameter"),
    ("current",        "Parameter"),
    ("frequency",      "Parameter"),
    ("baud",           "Parameter"),
    ("interface",      "Interface"),
    ("connector",      "Interface"),
    ("port",           "Interface"),
    ("software",       "Software"),
    ("firmware",       "Software"),
    ("library",        "Software"),
    ("package",        "Package"),
    ("topic",          "Topic"),
    ("node",           "Node"),
    ("driver",         "Module"),
    ("bridge",         "Module"),
    ("regulator",      "Module"),
    ("document",       "Document"),
    ("datasheet",      "Document"),
    ("manual",         "Document"),
    ("sensor",         "Component"),
    ("actuator",       "Component"),
    ("motor",          "Component"),
    ("robot",          "Component"),
    ("module",         "Module"),
]


# ── Relation type aliases ─────────────────────────────────────────────────────
# Maps raw LLM relation strings to canonical RELATION_TYPES.
# Applied after uppercasing and replacing spaces with underscores.

RELATION_ALIASES: dict[str, str] = {
    "IS_A":                  "IS_TYPE",
    "IS_AN":                 "IS_TYPE",
    "IS_TYPE_OF":            "IS_TYPE",
    "TYPE_OF":               "IS_TYPE",
    "DEFINED_AS":            "IS_TYPE",
    "IS_DEFINED_AS":         "IS_TYPE",
    "HAS":                   "HAS_PARAMETER",
    "HAS_PROPERTY":          "HAS_PARAMETER",
    "HAS_ATTRIBUTE":         "HAS_PARAMETER",
    "HAS_SPEC":              "HAS_PARAMETER",
    "HAS_SPECIFICATION":     "HAS_PARAMETER",
    "HAS_FEATURE":           "HAS_PARAMETER",
    "HAS_MEMORY":            "HAS_PARAMETER",
    "CONTAINS":              "PART_OF",
    "INCLUDED_IN":           "PART_OF",
    "COMPONENT_OF":          "PART_OF",
    "SUBSET_OF":             "PART_OF",
    "USES":                  "COMMUNICATES_VIA",
    "USES_PROTOCOL":         "COMMUNICATES_VIA",
    "COMMUNICATES_WITH":     "COMMUNICATES_VIA",
    "CONNECTS":              "CONNECTS_TO",
    "CONNECTED_TO":          "CONNECTS_TO",
    "WIRED_TO":              "CONNECTS_TO",
    "LINKED_TO":             "CONNECTS_TO",
    "CONTROLS_VIA":          "CONTROLS",
    "DRIVES":                "CONTROLS",
    "MANAGES":               "CONTROLS",
    "OPERATES":              "CONTROLS",
    "READS_FROM":            "READS",
    "RECEIVES_FROM":         "READS",
    "RECEIVES":              "READS",
    "WRITES_TO":             "WRITES",
    "SENDS_TO":              "WRITES",
    "SENDS":                 "WRITES",
    "TRANSMITS":             "WRITES",
    "PUBLISHES":             "PUBLISHES_TO",
    "PUBLISHED_TO":          "PUBLISHES_TO",
    "SUBSCRIBES":            "SUBSCRIBES_TO",
    "DEPENDS_ON":            "REQUIRES",
    "NEEDS":                 "REQUIRES",
    "NEEDS_LIBRARY":         "REQUIRES",
    "PRODUCES":              "GENERATES",
    "OUTPUTS":               "GENERATES",
    "GENERATES_SIGNAL":      "GENERATES",
    "RUNS_AT":               "OPERATES_AT",
    "CLOCKED_AT":            "OPERATES_AT",
    "POWERED_BY":            "OPERATES_AT",
    "CONFIGURED_VIA":        "CONFIGURED_BY",
    "SET_BY":                "CONFIGURED_BY",
    "ACTIVATED_BY":          "TRIGGERED_BY",
    "ENABLED_BY":            "TRIGGERED_BY",
    "TRIGGERED_VIA":         "TRIGGERED_BY",
    "INTERFACES":            "INTERFACES_WITH",
    "INTERACTS_WITH":        "INTERFACES_WITH",
    "WORKS_WITH":            "INTERFACES_WITH",
    "EQUIPPED_WITH":         "HAS_PIN",
    "HAS_CHANNEL":           "HAS_PIN",
    "HAS_PWM_PIN":           "HAS_PIN",
    "HAS_PERIPHERAL":        "PART_OF",
    "IS_EQUIPPED_WITH":      "HAS_PIN",
    "RELATED_TO":            "RELATED_TO",
}


# ── Protected hardware identifier patterns ────────────────────────────────────
# Entity names matching these patterns must NEVER be merged with other entities
# even if they appear similar. Validated in kg_validator.py.

PROTECTED_PATTERNS: list[re.Pattern] = [
    # ADC channels
    re.compile(r"^ADC[0-7]$", re.IGNORECASE),
    # Timers
    re.compile(r"^Timer[0-9]$", re.IGNORECASE),
    re.compile(r"^TC[0-9]$", re.IGNORECASE),
    # PWM output compare registers
    re.compile(r"^OC[0-9][AB]$", re.IGNORECASE),
    re.compile(r"^OCR[0-9][AB]?$", re.IGNORECASE),
    # Timer control registers
    re.compile(r"^TCCR[0-9][AB]$", re.IGNORECASE),
    re.compile(r"^TCNT[0-9]$", re.IGNORECASE),
    re.compile(r"^ICR[0-9]$", re.IGNORECASE),
    # Arduino digital pins
    re.compile(r"^D[0-9]{1,2}$", re.IGNORECASE),
    # Arduino analog pins
    re.compile(r"^A[0-7]$", re.IGNORECASE),
    # Interrupts
    re.compile(r"^INT[0-9]$", re.IGNORECASE),
    re.compile(r"^PCINT[0-9]{1,2}$", re.IGNORECASE),
    # SPI pins
    re.compile(r"^(MOSI|MISO|SCK|SS)$", re.IGNORECASE),
    # I2C pins
    re.compile(r"^(SDA|SCL)$", re.IGNORECASE),
    # UART pins
    re.compile(r"^(TX|RX|TXD|RXD)$", re.IGNORECASE),
    # Port registers
    re.compile(r"^(DDR|PORT|PIN)[A-F]$", re.IGNORECASE),
    # ATmega specific registers
    re.compile(r"^(ADMUX|ADCSRA|ADCSRB|ADCL|ADCH)$", re.IGNORECASE),
    re.compile(r"^(SREG|SPH|SPL|EECR|EEDR|EEARL|EEARH)$", re.IGNORECASE),
    re.compile(r"^(MCUCR|MCUSR|SMCR|PRR)$", re.IGNORECASE),
    re.compile(r"^(WDTCSR|CLKPR|OSCCAL|RCSTA)$", re.IGNORECASE),
]


def is_protected(entity_name: str) -> bool:
    """Return True if entity_name matches any protected pattern."""
    for pattern in PROTECTED_PATTERNS:
        if pattern.match(entity_name.strip()):
            return True
    return False


# ── Generic stopword entities to discard ─────────────────────────────────────
# Extracted entity names that match these are discarded by the validator.

GENERIC_STOPWORDS = {
    "system", "it", "this", "that", "the", "a", "an", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "thing", "things", "item", "items",
    "data", "value", "values", "type", "types", "mode", "modes",
    "function", "functions", "feature", "features", "property",
    "properties", "operation", "operations", "example", "examples",
    "user", "users", "output", "input", "method", "methods",
    "process", "processes", "result", "results", "information",
    "description", "text", "number", "bit", "byte", "word",
    "hardware", "software", "concept",  # too generic
}


# ── Ontology mapper ───────────────────────────────────────────────────────────

class OntologyMapper:
    """
    Normalises raw LLM entity types and relation types to canonical ontology.

    Usage:
        mapper = OntologyMapper()
        canonical_type = mapper.map_entity_type("timer counter")
        canonical_rel  = mapper.map_relation_type("IS_A")
    """

    def map_entity_type(self, raw_type: str) -> str:
        """
        Map raw LLM entity type to canonical ENTITY_TYPES member.
        Priority: exact alias → fuzzy keyword → default.
        """
        if not raw_type:
            return DEFAULT_ENTITY_TYPE

        # If already canonical, return as-is
        if raw_type in ENTITY_TYPES:
            return raw_type

        normalised = raw_type.strip().lower()

        # Exact alias lookup
        if normalised in TYPE_ALIASES:
            return TYPE_ALIASES[normalised]

        # Fuzzy keyword matching
        for keyword, canonical in FUZZY_RULES:
            if keyword in normalised:
                return canonical

        return DEFAULT_ENTITY_TYPE

    def map_relation_type(self, raw_relation: str) -> str:
        """
        Map raw LLM relation string to canonical RELATION_TYPES member.
        Uppercases, replaces spaces/hyphens with underscores, then aliases.
        """
        if not raw_relation:
            return DEFAULT_RELATION_TYPE

        # If already canonical, return as-is
        if raw_relation in RELATION_TYPES:
            return raw_relation

        # Normalise: uppercase, spaces/hyphens → underscore, strip non-alnum/_
        normalised = re.sub(r"[^A-Z0-9_]", "_",
                            raw_relation.upper().replace(" ", "_").replace("-", "_"))
        normalised = re.sub(r"_+", "_", normalised).strip("_")

        # Direct alias lookup
        if normalised in RELATION_ALIASES:
            return RELATION_ALIASES[normalised]

        # If normalised form is already a canonical type
        if normalised in RELATION_TYPES:
            return normalised

        # Fuzzy: check if any canonical type is a substring
        for canonical in RELATION_TYPES:
            if canonical in normalised or normalised in canonical:
                return canonical

        return DEFAULT_RELATION_TYPE

    def is_valid_entity_name(self, name: str) -> bool:
        """
        Return True if entity name is acceptable.
        Rejects: empty, too short, pure numbers, generic stopwords.
        """
        if not name or not name.strip():
            return False
        stripped = name.strip()
        if len(stripped) < 2:
            return False
        if stripped.lower() in GENERIC_STOPWORDS:
            return False
        # Reject pure numeric strings (e.g. "1", "0")
        if re.match(r"^\d+$", stripped):
            return False
        return True

    def is_valid_relation(
        self, source: str, relation: str, target: str
    ) -> bool:
        """
        Return True if relation triple is structurally valid.
        Rejects: missing fields, self-loops (unless protected terms).
        """
        if not source or not relation or not target:
            return False
        if source.strip().lower() == target.strip().lower():
            # Allow self-loops only for protected entities (rare edge case)
            return False
        return True
