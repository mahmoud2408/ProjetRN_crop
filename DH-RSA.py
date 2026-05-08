import os
import base64
import secrets
from hashlib import sha256

SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
]
INV_SBOX = [0] * 256
for i, v in enumerate(SBOX):
    INV_SBOX[v] = i

RCON = [
    0x00000000, 0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000, 0x1b000000, 0x36000000
]

def rot_word(w):
    return ((w << 8) & 0xffffffff) | (w >> 24)

def sub_word(w):
    return (
        (SBOX[(w >> 24) & 0xff] << 24) |
        (SBOX[(w >> 16) & 0xff] << 16) |
        (SBOX[(w >> 8) & 0xff] << 8) |
        SBOX[w & 0xff]
    )

def bytes_to_words(b):
    return [int.from_bytes(b[i:i+4], "big") for i in range(0, len(b), 4)]

def words_to_bytes(words):
    return b"".join(w.to_bytes(4, "big") for w in words)

def gf_mul(a, b):
    res = 0
    for _ in range(8):
        if b & 1:
            res ^= a
        hi = a & 0x80
        a = (a << 1) & 0xff
        if hi:
            a ^= 0x1b
        b >>= 1
    return res

def key_expansion(key):
    assert len(key) in (16, 24, 32)
    Nk = len(key) // 4
    Nr = Nk + 6
    w = bytes_to_words(key)
    for i in range(Nk, 4 * (Nr + 1)):
        temp = w[i - 1]
        if i % Nk == 0:
            temp = sub_word(rot_word(temp)) ^ RCON[i // Nk]
        elif Nk > 6 and i % Nk == 4:
            temp = sub_word(temp)
        w.append(w[i - Nk] ^ temp)
    return w, Nr

def round_key_bytes(expanded_words, round_idx):
    return words_to_bytes(expanded_words[4 * round_idx:4 * (round_idx + 1)])

def add_round_key(state, rk):
    for i in range(16):
        state[i] ^= rk[i]

def sub_bytes(state):
    for i in range(16):
        state[i] = SBOX[state[i]]

def inv_sub_bytes(state):
    for i in range(16):
        state[i] = INV_SBOX[state[i]]

def shift_rows(s):
    t = s[:]
    s[0], s[4], s[8], s[12] = t[0], t[4], t[8], t[12]
    s[1], s[5], s[9], s[13] = t[5], t[9], t[13], t[1]
    s[2], s[6], s[10], s[14] = t[10], t[14], t[2], t[6]
    s[3], s[7], s[11], s[15] = t[15], t[3], t[7], t[11]

def inv_shift_rows(s):
    t = s[:]
    s[0], s[4], s[8], s[12] = t[0], t[4], t[8], t[12]
    s[1], s[5], s[9], s[13] = t[13], t[1], t[5], t[9]
    s[2], s[6], s[10], s[14] = t[10], t[14], t[2], t[6]
    s[3], s[7], s[11], s[15] = t[7], t[11], t[15], t[3]

def mix_columns(s):
    for c in range(4):
        i = 4 * c
        a0, a1, a2, a3 = s[i:i+4]
        s[i+0] = gf_mul(a0, 2) ^ gf_mul(a1, 3) ^ a2 ^ a3
        s[i+1] = a0 ^ gf_mul(a1, 2) ^ gf_mul(a2, 3) ^ a3
        s[i+2] = a0 ^ a1 ^ gf_mul(a2, 2) ^ gf_mul(a3, 3)
        s[i+3] = gf_mul(a0, 3) ^ a1 ^ a2 ^ gf_mul(a3, 2)

def inv_mix_columns(s):
    for c in range(4):
        i = 4 * c
        a0, a1, a2, a3 = s[i:i+4]
        s[i+0] = gf_mul(a0, 14) ^ gf_mul(a1, 11) ^ gf_mul(a2, 13) ^ gf_mul(a3, 9)
        s[i+1] = gf_mul(a0, 9) ^ gf_mul(a1, 14) ^ gf_mul(a2, 11) ^ gf_mul(a3, 13)
        s[i+2] = gf_mul(a0, 13) ^ gf_mul(a1, 9) ^ gf_mul(a2, 14) ^ gf_mul(a3, 11)
        s[i+3] = gf_mul(a0, 11) ^ gf_mul(a1, 13) ^ gf_mul(a2, 9) ^ gf_mul(a3, 14)

def aes_encrypt_block(block, key):
    w, Nr = key_expansion(key)
    state = list(block)
    add_round_key(state, list(round_key_bytes(w, 0)))
    for rnd in range(1, Nr):
        sub_bytes(state)
        shift_rows(state)
        mix_columns(state)
        add_round_key(state, list(round_key_bytes(w, rnd)))
    sub_bytes(state)
    shift_rows(state)
    add_round_key(state, list(round_key_bytes(w, Nr)))
    return bytes(state)

def aes_decrypt_block(block, key):
    w, Nr = key_expansion(key)
    state = list(block)
    add_round_key(state, list(round_key_bytes(w, Nr)))
    for rnd in range(Nr - 1, 0, -1):
        inv_shift_rows(state)
        inv_sub_bytes(state)
        add_round_key(state, list(round_key_bytes(w, rnd)))
        inv_mix_columns(state)
    inv_shift_rows(state)
    inv_sub_bytes(state)
    add_round_key(state, list(round_key_bytes(w, 0)))
    return bytes(state)

def pkcs7_pad(data, block_size=16):
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len]) * pad_len

def pkcs7_unpad(data, block_size=16):
    if not data or len(data) % block_size != 0:
        raise ValueError("Longueur de donnée invalide.")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > block_size:
        raise ValueError("Padding invalide.")
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise ValueError("Padding invalide.")
    return data[:-pad_len]

def aes_cbc_encrypt(plaintext, key, iv):
    plaintext = pkcs7_pad(plaintext, 16)
    prev = iv
    out = b""
    for i in range(0, len(plaintext), 16):
        block = plaintext[i:i+16]
        x = bytes(a ^ b for a, b in zip(block, prev))
        c = aes_encrypt_block(x, key)
        out += c
        prev = c
    return out

def aes_cbc_decrypt(ciphertext, key, iv):
    if len(ciphertext) % 16 != 0:
        raise ValueError("Longueur du ciphertext invalide.")
    prev = iv
    out = b""
    for i in range(0, len(ciphertext), 16):
        block = ciphertext[i:i+16]
        p = aes_decrypt_block(block, key)
        out += bytes(a ^ b for a, b in zip(p, prev))
        prev = block
    return pkcs7_unpad(out, 16)



def is_probable_prime(n, rounds=16):
    if n < 2:
        return False

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for __ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_prime(bits):
    while True:
        n = secrets.randbits(bits)
        n |= (1 << (bits - 1)) | 1
        if is_probable_prime(n):
            return n

def generate_dh_parameters(bits=128):

    while True:
        q = generate_prime(bits - 1)
        p = 2 * q + 1
        if is_probable_prime(p):
            g = 2
            # Vérifie que g est bien générateur pour ce groupe
            if pow(g, (p - 1) // 2, p) != 1 and pow(g, (p - 1) // q, p) != 1:
                return p, g

def generate_private_key(p):
    return secrets.randbelow(p - 3) + 2

def generate_public_key(p, g, private_key):
    return pow(g, private_key, p)

def compute_shared_secret(their_public_key, private_key, p):
    return pow(their_public_key, private_key, p)

def derive_aes_256_key(shared_secret):

    secret_bytes = shared_secret.to_bytes((shared_secret.bit_length() + 7) // 8 or 1, "big")
    return sha256(secret_bytes).digest()


def encrypt_message(message, aes_key):
    iv = os.urandom(16)
    ciphertext = aes_cbc_encrypt(message.encode("utf-8"), aes_key, iv)
    return {
        "iv": base64.b64encode(iv).decode("utf-8"),
        "ciphertext": base64.b64encode(ciphertext).decode("utf-8")
    }

def decrypt_message(encrypted_data, aes_key):
    iv = base64.b64decode(encrypted_data["iv"])
    ciphertext = base64.b64decode(encrypted_data["ciphertext"])
    plaintext = aes_cbc_decrypt(ciphertext, aes_key, iv)
    return plaintext.decode("utf-8")


def main():
    print("=== Partie 1 : Diffie-Hellman ===")

    P, G = generate_dh_parameters(bits=128)
    print("P =", P)
    print("G =", G)

    alice_private = generate_private_key(P)
    alice_public = generate_public_key(P, G, alice_private)

    bob_private = generate_private_key(P)
    bob_public = generate_public_key(P, G, bob_private)

    print("Clé privée Alice =", alice_private)
    print("Clé publique Alice =", alice_public)
    print("Clé privée Bob    =", bob_private)
    print("Clé publique Bob   =", bob_public)

    alice_secret = compute_shared_secret(bob_public, alice_private, P)
    bob_secret = compute_shared_secret(alice_public, bob_private, P)

    print("Secret partagé Alice =", alice_secret)
    print("Secret partagé Bob   =", bob_secret)

    if alice_secret != bob_secret:
        print("Erreur : les secrets ne sont pas identiques.")
        return

    print("\n=== Partie 2 : Dérivation AES ===")
    aes_key = derive_aes_256_key(alice_secret)
    print("Clé AES-256 (base64) =", base64.b64encode(aes_key).decode("utf-8"))

    print("\n=== Chiffrement / Déchiffrement ===")
    message = "Bonjour, ceci est un message confidentiel."
    print("Message clair =", message)

    encrypted = encrypt_message(message, aes_key)
    print("IV        =", encrypted["iv"])
    print("Ciphertext =", encrypted["ciphertext"])

    decrypted = decrypt_message(encrypted, aes_key)
    print("Message déchiffré =", decrypted)

    assert decrypted == message
    print("\nSuccès : le message original a été retrouvé.")

if __name__ == "__main__":
    main()