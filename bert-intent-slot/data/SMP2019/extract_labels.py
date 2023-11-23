import json

if __name__ == '__main__':
    with open('data.json', 'r', encoding="utf-8") as f:
        data = json.load(f)

    # LAUNCH 102
    # QUERY 1598
    # ROUTE 74
    # SENDCONTACTS 23
    # SEND 101
    # REPLY 20
    # REPLAY_ALL 6
    # LOOK_BACK 6
    # NUMBER_QUERY 29
    # POSITION 40
    # PLAY 284
    # DEFAULT 15
    # DIAL 82
    # TRANSLATION 97
    # OPEN 75
    # CREATE 13
    # FORWARD 5
    # VIEW 1
    # SEARCH 8
    # RISERATE_QUERY 8
    # DOWNLOAD 2
    # DATE_QUERY 1
    # CLOSEPRICE_QUERY 2

    intent_labels = ['[UNK]']
    slot_labels = ['[PAD]','[UNK]', '[O]']
    for item in data:
        if item['intent'] not in intent_labels:
            intent_labels.append(item['intent'])

        for slot_name, slot_value in item['slots'].items():
            if 'B_'+slot_name not in slot_labels:
                slot_labels.extend(['I_'+slot_name, 'B_'+slot_name])

    with open('slot_labels.txt', 'w') as f:
        f.write('\n'.join(slot_labels))

    with open('intent_labels.txt', 'w') as f:
        f.write('\n'.join(intent_labels))
