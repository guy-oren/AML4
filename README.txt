Submitted by:

Guy Oren, ID 302764956, EMAIL: guyoren347@gmail.com
Shahar Azulay, ID 039764063, EMAIL: shahar4@gmail.com
Eitan-Hai Mashiah, ID: 206349045, EMAIL: eitanhaimashiah@gmail.com

Requirements:
    - Python 3+
    - numpy
    - pandas
    - pickle
    - scipy
    - networkx

Run:
    - python main.py --vocab_size <number>
    - NOTE: The argument vocab_size is optional, the default size is 400
    - NOTE: The data folder contains additional pickle file which is the result of processing label annotations for
      images. If the file does not exists, the code will create it
    - NOTE: Apparently there exist different machine codes with the same display name. To handle this we chose to number
      the duplications and still treat each machine code as different annotation
