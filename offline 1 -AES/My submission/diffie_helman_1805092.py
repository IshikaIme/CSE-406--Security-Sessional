import random
import time

def modular_pow(x, y, n):
    result = 1
    while y > 0:
        if y % 2 == 1:
            result = (result * x) % n
        y = y // 2
        x = (x * x) % n
    return result

def miillerTest(d, n):
     
    # Pick a random number in [2..n-2]
    # Corner cases make sure that n > 4
    a = 2 + random.randint(1, n - 4) 
 
    # Compute a^d % n
    x = modular_pow(a, d, n) 
 
    if (x == 1 or x == n - 1):
        return True 
 
    while (d != n - 1):
        x = (x * x) % n 
        d *= 2 
 
        if (x == 1):
            return False 
        if (x == n - 1):
            return True 
    return False 
 
def is_prime( n):
    k=5
    # Corner cases
    if (n <= 1 or n == 4):
        return False 
    if (n <= 3):
        return True 
    d = n - 1 
    while (d % 2 == 0):
        d //= 2 
 
    for i in range(k):
        if (miillerTest(d, n) == False):
            return False 
 
    return True 


def diffie_hellman(k, p, g):
    # Generate private key
    start_time_a = time.time()
    min= 1 << (k//2-1)
    while True:
        a = random.randint(min, p - 2)
        if is_prime(a)== True:
            break
    end_time_a = time.time()
    # Compute public key
    start_time_A = time.time()
    A = modular_pow(g, a, p)
    end_time_A = time.time()
    elapsed_time_a= end_time_a- start_time_a
    elapsed_time_A= end_time_A- start_time_A
    return a, A , elapsed_time_a, elapsed_time_A

def compute_shared_secret(public_key, private_key, p):
    # Compute shared secret
    start_time_ss = time.time()
    shared_secret = modular_pow(public_key, private_key, p)
    end_time_ss = time.time()
    elapsed_time_ss= end_time_ss- start_time_ss
    return shared_secret , elapsed_time_ss

def main():
    user_input = input("Enter an integer: ")
    k = int(user_input)
    start_time_p = time.time()
    p = 1 << (k-1)
    while is_prime(p)!= True and is_prime((p-1)//2)!=True:
        p+=1
    end_time_p = time.time()

    start_time_g = time.time()
    while True: 
        g = random.randint(2, p - 2)
        if (modular_pow(g, 2, p) )!=1 and (modular_pow(g, (p-1)//2, p) )!= 1:
            break
    end_time_g = time.time()
    # Alice generates keys
    alice_private_key, alice_public_key, elapsed_time_a, elapsed_time_A = diffie_hellman(k,p, g)
    print("a = ", alice_private_key)
   
    # Bob generates keys
    bob_private_key, bob_public_key, elapsed_time_b, elapsed_time_B = diffie_hellman(k,p, g)
    print("b = ", bob_private_key)

    # Alice and Bob exchange public keys and compute shared secret
   
    alice_shared_secret, elapsed_time_ss = compute_shared_secret(bob_public_key, alice_private_key, p)
    bob_shared_secret, elapsed_time_ss  = compute_shared_secret(alice_public_key, bob_private_key, p)
    

    # The shared secret should be the same for Alice and Bob
 
    print("Shared Secret (Alice):", alice_shared_secret)
    print("Shared Secret (Bob):", bob_shared_secret)

    print("elapsed time for p: ", end_time_p - start_time_p)
    print("elapsed time for g: ", end_time_g - start_time_g)
    print("elapsed time for a: ", elapsed_time_a)
    print("elapsed time for A: ", elapsed_time_A)
    print("elapsed time for b: ", elapsed_time_b)
    print("elapsed time for B: ", elapsed_time_B)
    print("elapsed time for shared secret: ", elapsed_time_ss)

main()