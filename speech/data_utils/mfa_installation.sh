mkdir mfa_input
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa align mfa_input english_us_arpa english_us_arpa mfa_output
