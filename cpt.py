# Get current cryptocurrency prices and update the README.md

from urllib import request
import os, json, re

CMC_API_KEY = os.environ['CMC_API_KEY']
CURRENCIES  = os.environ['CURRENCIES'].split(',')
SOURCE_PATH = 'README.md'

with open(SOURCE_PATH, 'r') as f:
    source = f.read()

for symbol in CURRENCIES:
    try:
        req = request.urlopen(request.Request(
            f'https://pro-api.coinmarketcap.com/v2/tools/price-conversion?amount=1&symbol={symbol}',
            headers={'Accept': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
        ))
    except Exception as e:
        print(e)
        os.exit(1)
    
    source = source.replace(re.search(f'<td id="#{symbol}">(.*?)<', source).groups()[0],
                            f'{json.loads(req.read())["data"][0]["quote"]["USD"]["price"]:,.2f}')

with open(SOURCE_PATH, 'w') as f:
    f.write(source)

