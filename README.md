this is basically a ai based system which extracts story texts from the manga pages, input was basically 3 consecutive manga pages and output is text lines in with speaker label all in JSONL format.
so first i connected google drive with my colab, than installed some libraries transformer to load and use Quen model, acclerate helps in excution of model qwen-vl-utils helps in processing image inputs tqdm to show progress bar in loops and -q for the quiet mode
imported os for files, pytorch library, Image to process images  configured the ROOT and OUTPUT directory for storing 
json data is loaded than , if the split field is development than it is stored in dev list similary for the test case basically list comprehension than  a dictionary is made , main diff between json load and json loads is json.load read json from file object while json loads reads data from json string 
eg - labels["seq_1"] = [
    [{"speaker": "character1", "text": "Hello!"}],
    [],
    [{"speaker": "NARRATION", "text": "The next day."}]
]
labels[seq1] will now give corect answers of that sequence . so development labels can be accessed through sequence id that is the jist
so all this was basically organization of data up until now.

so now we use our ai worker here, first the vision language model is uploaded and autoprocessor processes components of it we use 3 billion parametrized model, it is instruction based designed version, pretrained model is imported, to store numbers 16 bit floating numbers it decreases GPU memory requirement, CUDA GPU was specified to load , now processor it prepares with respect to required format, than we limit the size of images 

ai worker was given the instructions, now the get_page_text function processore prepares for both  promopt + image than   to qwen model and lastly output, messages in format were given like
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": PROMPT}
]}]
this tells model that this is the input, now the model needs specific format template for conversion it is done by apply_chat_template , than the images and texts were processed and stored in tensors, and than model is generated finally gradient is not calculated as we just want predictions not train the model, and greedy decoding is used in which highest probablity next token is selected
only the generated ans is written,   sliced token eg if the input gives 100 token and output returns 125 tokens than only the 25 tokens will be returned which are at last as tehy contain our answers
at last unused cache memory is released and as we want output in json format we parse the output by parse_lines function an empty list was created where parsed sentences will be contained, white spaces stripped, matched with regex format if matched than grp1 contain speaker labels while grp2 labels dialogue text
{"speaker": "character1", "text": "Hello!"}
{"speaker": "NARRATION", "text": "Later that day..."}
the output is dictionary list of both
conversion is done with to respect to structured records
did a sanity test on model

model answers are read line by line extra spaces are trimmed, if line is empty skip it go ahead all this is happening in the output of the model also checking the colons in the sentencesm in this parse_lines() if we could not get speaker name than default name is character1, so all this is happening before fine tuning, in rows list only saved and new answers are stored, and in done list sequence ids whose work is already done these are actually sets so tehy will store unique values only, 
advantage is just basically  that if colab disconnects i know which sequences are done and which are not; first i take 20 development sequences, result is stored in pages, if the error is in the model the sid tells us which sequence exactly has the problem, sequence id and its 3 pages are recorded and stored in rows than we just convert all this into jsonl format
this code actually rewrites the whole sequence in rows after 1 sequence is completed, but this is inefficient for big files
in error handling loop stops and the sequence id which had error is shown so the empty output is not saved, 

now we have the answers so now we compare the model answers with the actual answers, first 20 dev sequences IDs is erupted out, path of the file with correct answers is made, selected sequence id and for its correct answers the reference file is made, 
