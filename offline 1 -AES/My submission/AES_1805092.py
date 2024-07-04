from BitVector import *
import numpy as np
import time

ROW= 4
round_for_bits = {128 : 10, 192 : 12, 256 : 14}
AES_modulus = BitVector(bitstring='100011011') # Used in gf_multiply_modular()
Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

InvSbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

Mixer = [
    [BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03")],
    [BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02")]
]

InvMixer = [
    [BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09")],
    [BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D")],
    [BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B")],
    [BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E")]
]

#round key generation
round_constant = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]

def get_column(matrix, col_no):
    return [row[col_no] for row in matrix]

def convert_BitVector_list_to_int_list(bitvector_list):
    int_list = []
    for bitvector in bitvector_list:
        int_list.append(int(bitvector))
    return int_list

def matrix_to_hex(matrix):
    hex_str = ""
    for row in matrix:
        for element in row:
            hex_str += element.get_bitvector_in_hex()
    return hex_str

def transpose_matrix(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Create a new matrix with swapped dimensions
    transposed_matrix = [[0 for _ in range(num_rows)] for _ in range(num_cols)]

    # Transpose the matrix
    for i in range(num_rows):
        for j in range(num_cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

def matrix_to_hex_col_wise(matrix):
    trans_mat= transpose_matrix(matrix)
    hex_str = ""
    for row in trans_mat:
        for element in row:
            hex_str += element.get_bitvector_in_hex()
    return hex_str

def matrix_to_hex_list(matrix):
    hex_list = []
    for row in matrix:
        for element in row:
            hex_list.append(element.get_bitvector_in_hex())
    return hex_list

def print_hex_from_int(intList):
    hex_list = []
    for i in intList:
        hex_list.append(hex(i))
    return hex_list

def generate_round_key(prev_roundkey, round_no):
    # Split the previous round key into 4 words
    #prev_roundkey= matrix_to_hex(prev_roundkey)

    w0 = convert_BitVector_list_to_int_list(get_column(prev_roundkey, 0))
    w1 = convert_BitVector_list_to_int_list(get_column(prev_roundkey, 1))
    w2 = convert_BitVector_list_to_int_list(get_column(prev_roundkey, 2))
    w3 = convert_BitVector_list_to_int_list(get_column(prev_roundkey, 3))

    temp = w3
    m = g(w3, round_no)
    #print("m =", print_hex_from_int(m))
   #print( "w0 = ", print_hex_from_int(w0), " w1 = ", print_hex_from_int(w1), " w2 = ", print_hex_from_int(w2) , " w3 = ",print_hex_from_int(w3))
    w4 = xor_rows(w0 , m)
    w5 = xor_rows(w4 , w1)
    w6 = xor_rows(w5 , w2)
    w7 = xor_rows(w6 , temp)

    round_key_int_list = np.array(w4 + w5 + w6 + w7)
    round_key_int_mat = round_key_int_list.reshape(4, 4)
    round_key_int_mat = round_key_int_mat.transpose()


    #! convert to bitvector
    round_key_bitvector_mat = []
    for row in round_key_int_mat:
        round_key_bitvector_mat.append([BitVector(intVal=el, size=8) for el in row])
    return round_key_bitvector_mat

def g(col, round_no):
   # print("round no =",  round_no)
    r = left_shift(col)
    r = [Sbox[el] for el in r]
    r = xor_rows(r,[round_constant[round_no],00,00,00])
    #print( "r ", print_hex_from_int(r))
    return r


#code for substitute bytes
def sub_bytes_from_sbox(b):
    int_val = int(b)
    s = Sbox[int_val]
    s = BitVector(intVal=s, size=8)
    return s

def sub_bytes_row(row):
   # print("row" , row)
    st= convert_BitVector_list_to_int_list(row)
    b = []
    for r in st:
        s = sub_bytes_from_sbox( r )
        b.append(s)
    return b

def sub_bytes_row_hex(row):
   #  print("row" , row)
    b = []
    for char in row:
        s = sub_bytes_from_sbox(BitVector(hexstring= char))
        b.append(s)
    return b

def sub_bytes_matrix(mat):
    m = []
    for row in mat:
        m.append(sub_bytes_row(row))
    return m



#code for substitute bytes in decryption
def sub_bytes_from_inverse_sbox(b):
    int_val = int(b)
  #  print("int val = " , int_val)
    s = InvSbox[int_val]
    s = BitVector(intVal=s, size=8)
    return s


def inverse_sub_bytes_row(row):
   # print("row" , row)
    st= convert_BitVector_list_to_int_list(row)
    b = []
    for r in st:
        s = sub_bytes_from_inverse_sbox( r )
        b.append(s)
    return b

def inverse_sub_bytes_matrix(mat):
    m = []
    for row in mat:
        m.append(inverse_sub_bytes_row(row))
    return m

def inverse_sub_bytes_row_hex(row):
   #  print("row" , row)
    b = []
    for char in row:
        s = sub_bytes_from_inverse_sbox(BitVector(hexstring= char))
        b.append(s)
    return b


#code for right shift of nth row n times
def right_shift_row(matrix, row_index, num_shifts):
    row = matrix[row_index]
    shifted_row = row[-num_shifts:] + row[:-num_shifts]
    matrix[row_index] = shifted_row
    return matrix[row_index]

def right_shift_mat(matrix):
    for row_index in range(len(matrix)):
        matrix[row_index] = right_shift_row(matrix, row_index, row_index)
    return matrix


#code for left shift of nth row n times
def left_shift_row(matrix, row_index, num_shifts):
    row = matrix[row_index]
    shifted_row = row[num_shifts:] + row[:num_shifts]
    matrix[row_index] = shifted_row
    return matrix[row_index]

def left_shift_mat(matrix):
    for row_index in range(len(matrix)):
        matrix[row_index] = left_shift_row(matrix, row_index, row_index )
    return matrix


#code for normal left shift
def left_shift(row):
    shifted_row = row[1:] + row[:1]
    return shifted_row


#convert to hex
def convert_ascii_to_hex_array_row(text):
    cipher = ''.join(format(ord(i), '02x') for i in text)
    cipher = [cipher[i:i+2] for i in range(0, len(cipher), 2)]
    return cipher


#xor of 2 rows
def xor_rows(a,b):
    if len(a) != len(b):
        raise ValueError("Lists must have the same length")
    r = []
    length = len(a)
    for i in range(length):
        r.append(a[i] ^ b[i])
    return r

#xor of 2 matrix
def xor_matrices(matrix1, matrix2):
    result = []
    for i in range(len(matrix1)):
        row1 = matrix1[i]
        row2 = matrix2[i]
        xor_result = xor_rows(row1, row2)
        result.append(xor_result)

    return result


#process key
def key_process(key,length):
    byte_len = length//8
    if(len(key) > byte_len):
        key = key[:byte_len]
    key += '0' * ( 16 - len(key)) 
    return key


#AES encryption
def AES_decryption( cipher_text, key, filler, chunks, round_keys, length):
   # print(" cipehr = " , cipher_text)

   # print(" cipehr = " , cipher)
    cipher_text = convert_ascii_to_hex_array_row(cipher_text)
    #print(" cipehr = " , cipher_text)
    k = convert_ascii_to_hex_array_row(key)
    
    # process key
    key = key_process(k, length)
    block_size = length//8
    text_size = len(cipher_text) 
   

    for i in range(0, filler):
        cipher_text = cipher_text[:-2]

    print("DECRYPTION STARTED")
    
    
    arr=[]
   # print("text = ", text)
   # print("block size = ", block_size)
    for i in range(0, chunks):
        arr.append(decrypt(cipher_text[i*block_size : (i+1)*block_size], key,round_keys, length))
    return arr


def decrypt(text, key, round_keys, length):
    column = length // 32 
   # text=  cipher_text_to_hex(text) 
   # print(text)
    state_matrix = convert_to_col_order_array(text)
   # print(" state matrix = " , matrix_to_hex_list(state_matrix ))
 #   round_key_last = convert_to_col_order_array(key)
    total_rounds= len(round_keys)
    #print(" total rounds = ", total_rounds)
    round_key_last = round_keys[total_rounds-1]
    #print("st", state_matrix)
    #print("rk ", matrix_to_hex_list(round_key_last))
    state_matrix = xor_matrices(state_matrix, round_key_last)
    
    

    for i in range(1, round_for_bits[length]):
        state_matrix = right_shift_mat(state_matrix)
       # print(" state matrix = " , matrix_to_hex_list(state_matrix ))
        state_matrix = inverse_sub_bytes_matrix(state_matrix)
        state_matrix = xor_matrices(state_matrix,round_keys[total_rounds-i-1])
        state_matrix = inverse_mix_columns(column, state_matrix)
        
       # print(" state matrix = " , matrix_to_hex_list(state_matrix ))  

    #for last round
   # last= len(round_keys)
   # print("last ", last)
    #print(" jahmela =" ,matrix_to_hex_col_wise(round_key[last-1] ))
    state_matrix= right_shift_mat(state_matrix)
    state_matrix = inverse_sub_bytes_matrix(state_matrix)
   # print("len ", len(round_key))
    
    state_matrix = xor_matrices(state_matrix,round_keys[0])
   # print(" state matrix = " , matrix_to_hex_list(state_matrix ))

   # print("---------------------------------------------")
    '''
    for i in range(0, round_for_bits[length]+1):
        mat = matrix_to_hex_col_wise(round_key[i] )
        print(mat)
    print("---------------------------------------------")'''
    return state_matrix




#AES encryption
def AES_encryption( plain_text, given_key, length):
    text = convert_ascii_to_hex_array_row(plain_text)
    k = convert_ascii_to_hex_array_row(given_key)
    filler_count=0
    # process key
    key = key_process(k, length)
    #print("key  after process ", key)
    #count blocksize of text
    block_size = length//8
    text_size = len(text) 
    last_filled= 0
     

    if(text_size > block_size):
        if(text_size%block_size==0):
            chunk_numbers = text_size / block_size
            last_filled=1
            
        else:
            chunk_numbers = text_size / block_size + 1
            last_filled=0
    else:
        chunk_numbers =  1
        if(text_size==block_size):
            last_filled=1
        else:
            last_filled=0

        #fill with 00 to make it 16 bytes
    if(last_filled==0):
        filler_count = 0
        while((text_size + filler_count)%block_size != 0):
            text.append("00")
            filler_count += 1

    #print("chunk numbers = ", chunk_numbers)
    print("ENCRYPTION STARTED")
    arr = []
   # print("text = ", text)
   # print("block size = ", block_size)
    for i in range(0, chunk_numbers):
        st_time, round_key_stored, chunk_text = encrypt(text[i*block_size : (i+1)*block_size], key, length)
        arr.append(chunk_text)
    return st_time, round_key_stored, arr, filler_count, chunk_numbers
    

def encrypt(text, key, length):
    column = length // 32 
    state_matrix = convert_to_col_order_array(text)
    start_time_key = time.time()
    round_key0 = convert_to_col_order_array(key)
    state_matrix = xor_matrices(state_matrix, round_key0)
    round_key=[]
    round_key.append(round_key0)
    

    
    #print(" rkey = " , key )    
   # print(" state matrix = " , matrix_to_hex_list(state_matrix ))

    '''print("---------------------------------------------")
    
    for j in round_key[0]:
        for k in j:
            print(" r k for round ", 0 , " ", k.get_bitvector_in_hex() )
    print("---------------------------------------------")
'''
    #for round 1 to before last round 
    for i in range(1, round_for_bits[length]):
        state_matrix = left_shift_mat(state_matrix)
        state_matrix = sub_bytes_matrix(state_matrix)
        round_key.append(generate_round_key(round_key[i-1], i))
        state_matrix = mix_columns(column, state_matrix)
        state_matrix = xor_matrices(state_matrix,round_key[i])
       # print(" state matrix = " , matrix_to_hex_list(state_matrix ))  

    #for last round
    last= len(round_key)
   # print("last ", last)
    #print(" jahmela =" ,matrix_to_hex_col_wise(round_key[last-1] ))
    state_matrix= left_shift_mat(state_matrix)
    state_matrix = sub_bytes_matrix(state_matrix)
   # print("len ", len(round_key))
    round_key.append(generate_round_key(round_key[last-1], last))
    state_matrix = xor_matrices(state_matrix,round_key[last])
    scheduling_time = time.time()- start_time_key
   # print(" state matrix = " , matrix_to_hex_list(state_matrix ))

   # print("---------------------------------------------")
    '''
    for i in range(0, round_for_bits[length]+1):
        mat = matrix_to_hex_col_wise(round_key[i] )
        print(mat)
    print("---------------------------------------------")'''
    
    return scheduling_time,round_key, state_matrix



# A 2D list to contain the words in column-major order
def convert_to_col_order_array(hexArray):
    mat = [] 
    length = len(hexArray)
    for j in range(4):
        row = []
        for i in range(j, length, 4):
            row.append(BitVector(hexstring = hexArray[i]))
        mat.append(row)
    return mat


def convert_to_col_order_array_from_list(hexArray):
    mat = [] 
    length = len(hexArray)
    for j in range(4):
        row = []
        for i in range(j, length, 4):
            row.append(hexArray[i])
        mat.append(row)
    return mat

#mix_columns
def mix_columns(col, mat):
    ret = []
    for _ in range(ROW):
        ret.append([BitVector(intVal=0, size=8)] * col)

    for i in range(ROW):
        for j in range(col):
            for k in range(ROW):
                ret[i][j] ^= Mixer[i][k].gf_multiply_modular(mat[k][j], AES_modulus, 8)

    return ret

#mix_columns for decryption
def inverse_mix_columns(col, mat):
    ret = []
    for _ in range(ROW):
        ret.append([BitVector(intVal=0, size=8)] * col)

    for i in range(ROW):
        for j in range(col):
            for k in range(ROW):
                ret[i][j] ^= InvMixer[i][k].gf_multiply_modular(mat[k][j], AES_modulus, 8)

    return ret

# cipher text print
def cipher_text_to_hex(cr_text):
    ret = []
    for i in range(len(cr_text)):
        cryptic1D = convert_2D_To_1D(cr_text[i])
        toAdd = convert_To_Bit_Vector_Hex(cryptic1D)
        toAdd = ''.join(toAdd)
        ret.append(toAdd)
    return ret 

def convert_2D_To_1D(mat):
    ret = []
    R = len(mat)
    C = len(mat[0])
    for i in range(C):
        for j in range(R):
            ret.append(mat[j][i])
    return ret

def convert_To_Bit_Vector_Hex(vec):
    ret = []
    for i in range(len(vec)):
        ret.append(vec[i].get_bitvector_in_hex())
    return ret

def print_ascii(text):  
    str=''
    split_list = [text[i:i+2] for i in range(0, len(text), 2)]
    for ascii_value in split_list:
        #print((ascii_value))
        ascii_char = chr(int(ascii_value, 16))
        #print(ascii_char)
        str+= ascii_char
    return str

def hex_list_to_string(list):
    str=''
    for i in range(len(list)):
        str+= list[i]
    return str 


def main():
    key = "BUET CSE18 Batch"
    #text = "Two One Nine Two"
    text = "Can They Do This"
   # key = "Thats my Kung Fu"
   # print(key)
    size= 128


    
    print("Plain Text : ")
    print("In ASCII : ", text)
    print("In hexa : " , hex_list_to_string(convert_ascii_to_hex_array_row(text)))

    print("Key : ")
    print("In ASCII : ", key)
    print("In hexa : " , hex_list_to_string(convert_ascii_to_hex_array_row(key)))

    # do aes encryption on text
    start_time_en = time.time()
    scheduling_time, round_key_stored, cipher_text, filler_count, chunks =  AES_encryption(text , key, size)
    encryption_time = time.time() - start_time_en


    hex_cipher= cipher_text_to_hex(cipher_text)
    ascii_string = print_ascii(hex_list_to_string(hex_cipher))
    #print("len of asciistring ", len(ascii_string))
    
    print("Ciphered Text : ")
    print("In Hex :" , hex_list_to_string(hex_cipher))
    print("In ASCII:" , ascii_string )

    start_time_de = time.time()
    decrpytic = AES_decryption( ascii_string, key, filler_count,chunks,round_key_stored, size)
    decryption_time = time.time() - start_time_de
    
    hex_decipher= cipher_text_to_hex(decrpytic)

    print("De ciphered Text : ")
    print("In Hex :" , hex_list_to_string(hex_decipher))
   # print("In ASCII:" , ascii_destring)
    decipher_ascii = print_ascii(hex_list_to_string(hex_decipher))
    print("In ASCII:" , decipher_ascii)

    print("Execution Time Details: ")
    print("Key scheduling : " ,scheduling_time, " seconds" )
    print("Encryption time : " , encryption_time , " seconds")
    print("Decryption time : " , decryption_time , " seconds")


main()