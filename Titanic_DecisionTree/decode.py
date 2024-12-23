def decode(message_file):
    # Read the contents of the file
    with open(message_file, 'r') as file:
        lines = file.readlines()
    
    decoded_message = []
    for i, line in enumerate(lines):
        # Split each line into number and word
        number, word = line.strip().split(' ')
        number = int(number)
        # Add the word to the decoded message if its corresponding number is at the end of the pyramid line
        if number == (i + 1):
            decoded_message.append(word)
    
    # Join the decoded words into a string
    decoded_string = ' '.join(decoded_message)
    return decoded_string

# Example usage:
decoded_message = decode(r'C:\Users\DELL\OneDrive\Desktop\CSUS\CSC215\CSC215_Assg6_Ayush\coding_qual_input.txt')
print("Message",decoded_message)