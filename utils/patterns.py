"""
Sensitive data patterns and detection utilities
"""

SENSITIVE_PATTERNS = {
    # Contact Information
    "EMAIL": r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
    "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "PHONE_INTL": r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
    
    # Government IDs
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PASSPORT": r"\b[A-Z]{1,2}\d{6,9}\b",
    
    # Financial
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "BANK_ACCOUNT": r"\b\d{9,18}\b",
    "IFSC": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    "SWIFT": r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b",
    
    # Network & Security
    "IPV4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:/\d{1,2})?\b",
    "IPV6": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
    "MAC_ADDRESS": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b",
    "PORT": r":\d{2,5}\b(?=\s|,|$|\)|})",
    
    # AWS & Cloud
    "AWS_ACCESS_KEY": r"\b(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA|APKA)[A-Z0-9]{16}\b",
    "AWS_SECRET_KEY": r"\b[A-Za-z0-9/+=]{40}\b(?=\s|,|\"|}|$)",
    "AWS_ARN": r"\barn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d{12}:[a-zA-Z0-9\-/:]+",
    "AWS_ACCOUNT_ID": r"\b\d{12}(?=:|\"|\s|,|}|\))",
    "GOOGLE_API_KEY": r"\bAIza[0-9A-Za-z\-_]{35}\b",
    
    # Tokens & Keys
    "JWT_TOKEN": r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",
    "BEARER_TOKEN": r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*",
    "GITHUB_TOKEN": r"\bghp_[A-Za-z0-9]{36}\b",
    "GITHUB_OAUTH": r"\bgho_[A-Za-z0-9]{36}\b",
    "API_KEY": r"\b[A-Za-z0-9_\-]{32,45}\b",
    "PRIVATE_KEY": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[^-]+-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    
    # Database & Connection Strings
    "DB_CONNECTION": r"(?i)(?:mongodb|mysql|postgresql|jdbc|redis):\/\/[^\s\"'<>]+",
    "PASSWORD_FIELD": r"(?i)(?:password|passwd|pwd|secret)[\"\s]*[:=][\"\s]*[^\s\"}{,]{6,}",
    
    # Dates & IDs
    "DOB": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "DATE_ISO": r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))?\b",
    "VEHICLE": r"\b[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,2}[ -]?\d{4}\b",
    
    # Postal Codes
    "ZIPCODE_US": r"\b\d{5}(?:-\d{4})?\b",
    "POSTAL_CODE_UK": r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b",
    "PINCODE_INDIA": r"\b\d{6}\b",
    "POSTAL_CODE_CANADA": r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b",
    
    # Addresses
    "ADDRESS_STREET": r"\b\d+\s+(?:[A-Z][a-z]+\s+){1,5}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl|Parkway|Pkwy|Highway|Hwy|Terrace|Ter)\b",
    "ADDRESS_PO_BOX": r"\bP\.?O\.?\s*Box\s+\d+\b",
    "ADDRESS_UNIT": r"\b(?:Apt|Apartment|Suite|Unit|#)\s*[A-Z0-9\-]+\b",
    "ADDRESS_BUILDING": r"\b(?:Building|Block|Tower|Wing)\s+[A-Z0-9\-]+\b",
    "ADDRESS_FLOOR": r"\b(?:Floor|Flr|Level)\s+\d+\b",
    "ADDRESS_CITY_STATE": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\s+\d{5}\b",
    "ADDRESS_FULL_US": r"\d+\s+[A-Za-z0-9\s,]+,\s*[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}",
    "ADDRESS_INDIAN": r"\b(?:House|H\.?No\.?|Plot|Shop)\s*[#:]?\s*[\d\-/A-Z]+[,\s]+(?:[A-Z][a-z]+\s*){1,4}(?:Nagar|Colony|Road|Street|Area|Layout|Extension|Ext|Society|Sector)",
    
    # Names & Personal Info
    "NAME_AFTER_KEYWORD": r"(?i)(?:name|naam|नाम|student\s*name|employee\s*name|full\s*name)[\s:=-]+([A-Za-zा-ॿ\s]{2,50})",
    "FATHER_NAME": r"(?i)(?:father|पिता|father'?s?\s*name)[\s:=-]+([A-Za-zा-ॿ\s]{2,50})",
    "MOTHER_NAME": r"(?i)(?:mother|माता|mother'?s?\s*name)[\s:=-]+([A-Za-zा-ॿ\s]{2,50})",
    
    # Student/Employee IDs
    "STUDENT_ID": r"(?i)(?:student\s*id|student\s*no|roll\s*no|roll\s*number|enrollment\s*no)[\s:=-]+([A-Za-z0-9\-/]{3,20})",
    "EMPLOYEE_ID": r"(?i)(?:employee\s*id|emp\s*id|staff\s*id|employee\s*no|emp\s*no)[\s:=-]+([A-Za-z0-9\-/]{3,20})",
    "GENERIC_ID": r"(?i)(?:id\s*number|id\s*no|identification)[\s:=-]+([A-Za-z0-9\-/]{3,20})",
    "ALPHANUMERIC_ID_1": r"\b(?:EMP|STU|STUD|ROLL|ENR|ID)[A-Z]*\d{4,10}\b",
    "ALPHANUMERIC_ID_2": r"\b\d{2,4}[A-Z]{2,6}\d{4,10}\b",
    "ALPHANUMERIC_ID_3": r"\b[A-Z]{2,4}\d{6,10}\b",
    "ALPHANUMERIC_ID_4": r"\b\d{4,6}[A-Z]{1,3}\d{3,6}\b",
    
    # JSON/Structured Data
    "JSON_ROLE_ID": r'"RoleId"\s*:\s*"[A-Z0-9]+"',
    "JSON_ROLE_NAME": r'"RoleName"\s*:\s*"[^"]+"',
    "JSON_ARN": r'"Arn"\s*:\s*"arn:aws:[^"]+"',
    "JSON_SID": r'"Sid"\s*:\s*"[^"]+"',
    "JSON_AWS": r'"AWS"\s*:\s*"arn:aws:[^"]+"',
    "JSON_ACCOUNT": r'"AccountId"\s*:\s*"\d{12}"',
    "JSON_KEY_VALUE": r'"[A-Za-z]{4,}"\s*:\s*"[^"]{10,}"',
}

# Priority order for pattern matching
PRIORITY_PATTERNS = [
    "PRIVATE_KEY", "JWT_TOKEN", "BEARER_TOKEN", "AWS_ARN", 
    "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "GOOGLE_API_KEY",
    "GITHUB_TOKEN", "GITHUB_OAUTH", "DB_CONNECTION", "PASSWORD_FIELD",
    "JSON_ARN", "JSON_ROLE_ID", "JSON_ROLE_NAME", "JSON_SID", "JSON_AWS", "JSON_ACCOUNT",
    "EMAIL", "IPV4", "IPV6", "MAC_ADDRESS", 
    "AWS_ACCOUNT_ID", "AADHAAR", "PAN", "CREDIT_CARD", "SSN",
    "PHONE", "PHONE_INTL", "VEHICLE", "DATE_ISO", "DOB", 
    "ADDRESS_FULL_US", "ADDRESS_INDIAN", "ADDRESS_STREET", "ADDRESS_CITY_STATE",
    "ADDRESS_PO_BOX", "ADDRESS_UNIT", "ADDRESS_BUILDING", "ADDRESS_FLOOR",
    "ZIPCODE_US", "POSTAL_CODE_UK", "PINCODE_INDIA", "POSTAL_CODE_CANADA",
    "NAME_AFTER_KEYWORD", "FATHER_NAME", "MOTHER_NAME",
    "STUDENT_ID", "EMPLOYEE_ID", "GENERIC_ID",
    "ALPHANUMERIC_ID_1", "ALPHANUMERIC_ID_2", "ALPHANUMERIC_ID_3", "ALPHANUMERIC_ID_4"
]

# NER entity labels to redact
NER_LABELS = ["PERSON", "GPE", "ORG", "LOC", "FAC"]