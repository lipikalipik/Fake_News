import pandas as pd

# Sample fake news dataset with a balanced mix of real (0) and fake (1) news
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'title': [
        'Government plans to build a new bridge',
        'Aliens spotted in New York City!',
        'New study reveals the benefits of chocolate',
        'Scientists develop new energy source from water',
        'Fake news detection system released',
        'President announces new policy on education',
        'Mysterious creature found in the Amazon forest',
        'New breakthrough in cancer research',
        'Zombie apocalypse predicted for next year',
        'Mars mission successfully lands on the planet'
    ],
    'text': [
        'The government has announced a new plan to build a bridge...',
        'Several witnesses report seeing UFOs hovering over NYC...',
        'A recent study shows that consuming chocolate can improve health...',
        'A group of scientists has developed a revolutionary way to extract energy from water...',
        'A new system designed to detect fake news was launched today...',
        'The president has introduced a new education policy to help improve schools...',
        'A mysterious creature has been found deep in the Amazon forest...',
        'Scientists have made a significant breakthrough in the fight against cancer...',
        'A group of conspiracy theorists predict that a zombie apocalypse will occur next year...',
        'The Mars mission has successfully landed on the planet, making history...'
    ],
    'label': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 0 = Real, 1 = Fake
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv('fake_news.csv', index=False)

print("fake_news.csv created successfully!")
