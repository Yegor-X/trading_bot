import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import datetime as dt
import time
from optibook.synchronous_client import Exchange
from libs import print_positions_and_pnl, round_down_to_tick, round_up_to_tick

def train_optimized_classifier():
   df = pd.read_csv('training.csv')
   stocks = ['NVDA', 'ING', 'SAN', 'PFE', 'CSCO']
   
   def clean_text(text):
       text = str(text).lower()
       text = re.sub(r'@\w+:', '', text)
       text = re.sub(r'http\S+|www\S+', '', text)
       text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
       text = re.sub(r'\s+', ' ', text).strip()
       return text
   
   df['cleaned_text'] = df['SocialMediaFeed'].apply(clean_text)
   
   vectorizer = TfidfVectorizer(
       max_features=300,
       stop_words='english',
       ngram_range=(1, 2),
       min_df=2,
       max_df=0.95
   )
   
   X = vectorizer.fit_transform(df['cleaned_text'])
   
   classifiers = {}
   for stock in stocks:
       impacts = df[stock].abs()
       non_zero_impacts = impacts[impacts > 0]
       
       if len(non_zero_impacts) == 0:
           continue
           
       threshold = np.percentile(non_zero_impacts, 80)
       y = (impacts >= threshold).astype(int)
       
       if y.sum() < 5:
           continue
       
       clf = RandomForestClassifier(
           n_estimators=100,
           max_depth=10,
           min_samples_split=3,
           random_state=42,
           class_weight='balanced'
       )
       
       clf.fit(X, y)
       classifiers[stock] = clf
   
   return vectorizer, classifiers

def improved_keyword_detection(text, stock):
   text = text.lower()
   
   if stock == 'NVDA':
       patterns = [
           r'\bnvidia\b',
           r'\bnvda\b',
           r'\bgpu\b.*\b(nvidia|nvda)\b',
           r'\b(nvidia|nvda)\b.*\bgpu\b',
           r'\bgeforce\b',
           r'\brtx\b.*\b(graphics|gaming)\b',
           r'\bcuda\b',
           r'\btensor\b.*\bcore\b'
       ]
       return any(re.search(pattern, text) for pattern in patterns)
   
   elif stock == 'PFE':
       patterns = [
           r'\bpfizer\b',
           r'\bpfe\b',
           r'\bpfizer\b.*\b(drug|vaccine|pharmaceutical)\b',
           r'\b(drug|vaccine|pharmaceutical)\b.*\bpfizer\b',
           r'\bbiontech\b.*\bpfizer\b',
           r'\bpfizer\b.*\bbiontech\b'
       ]
       return any(re.search(pattern, text) for pattern in patterns)
   
   elif stock == 'ING':
       patterns = [
           r'\bing\s+bank\b',
           r'\bing\s+group\b',
           r'\bdutch\s+bank\b.*\bing\b',
           r'\bing\b.*\bdutch\s+bank\b',
           r'\bnetherlands\b.*\bing\b.*\bbank\b',
           r'\bing\s+groep\b',
           r'\bing\s+financial\b'
       ]
       if re.search(r'\bing\b', text):
           banking_context = re.search(r'\b(bank|banking|financial|credit|loan|mortgage)\b', text)
           direct_patterns = any(re.search(pattern, text) for pattern in patterns)
           return direct_patterns or (banking_context and re.search(r'\bing\s+(bank|group)\b', text))
       return False
   
   elif stock == 'SAN':
       patterns = [
           r'\bsantander\b',
           r'\bbanco\s+santander\b',
           r'\bspanish\s+bank\b.*\bsantander\b',
           r'\bsantander\b.*\bspanish\s+bank\b',
           r'\bspain\b.*\bsantander\b.*\bbank\b'
       ]
       return any(re.search(pattern, text) for pattern in patterns)
   
   elif stock == 'CSCO':
       patterns = [
           r'\bcisco\b',
           r'\bcsco\b',
           r'\bcisco\s+systems\b',
           r'\bnetworking\b.*\bcisco\b',
           r'\bcisco\b.*\bnetworking\b',
           r'\brouter\b.*\bcisco\b',
           r'\bcisco\b.*\brouter\b'
       ]
       return any(re.search(pattern, text) for pattern in patterns)
   
   return False

def predict_news_impact_improved(news_text, vectorizer, classifiers, ml_threshold=0.15):
   cleaned = str(news_text).lower()
   cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)
   
   X = vectorizer.transform([cleaned])
   
   affected_stocks = []
   
   for stock, clf in classifiers.items():
       ml_prob = clf.predict_proba(X)[0][1]
       keyword_match = improved_keyword_detection(news_text, stock)
       
       if keyword_match and ml_prob < 0.3:
           final_confidence = 0.6
           reason = "keyword"
       elif ml_prob >= ml_threshold:
           final_confidence = ml_prob
           reason = "ml"
       else:
           continue
       
       affected_stocks.append((stock, final_confidence, reason))
   
   return affected_stocks

class HackathonTradingBot:
   
   def __init__(self):
       self.exchange = Exchange()
       self.vectorizer = None
       self.classifiers = None
       self.paused_until = {}
       self.overlimit_start_time = None
       self.in_cooldown_until = None
       
       self.QUOTED_VOLUME = 20
       self.FIXED_CREDIT = 0.12
       self.POSITION_RETREAT = 0.003
       self.POSITION_LIMIT = 50
       self.TARGET_POSITION = 45
       
       self.CONFIDENCE_THRESHOLD = 0.15
       self.PAUSE_DURATION = 20
       self.MAX_NEWS_AGE = 60
       
       self.trades_count = 0
       self.news_reactions = 0
       self.start_time = dt.datetime.now()
   
   def setup(self):
       self.exchange.connect()
       self.instruments = self.exchange.get_instruments()
       self.vectorizer, self.classifiers = train_optimized_classifier()
       
       now = dt.datetime.now()
       for instrument_id in self.instruments:
           self.paused_until[instrument_id] = now

   def check_position_limits(self):
       total_position = 0
       for instrument_id, pos in self.exchange.get_positions().items():
           total_position += abs(pos)

       now = dt.datetime.now()

       if total_position > 50:
           if self.overlimit_start_time is None:
               self.overlimit_start_time = now
           elif (now - self.overlimit_start_time).total_seconds() > 55:
               self.in_cooldown_until = now + dt.timedelta(seconds=5)
       else:
           self.overlimit_start_time = None

       if self.in_cooldown_until and now < self.in_cooldown_until:
           return True

       return False
   
   def check_for_news(self):
       try:
           feeds = self.exchange.poll_new_social_media_feeds()
           
           if not feeds:
               return
           
           for feed in feeds:
               news_age = (dt.datetime.now() - feed.timestamp).total_seconds()
               if news_age > self.MAX_NEWS_AGE:
                   continue
               
               news_text = feed.post
               
               affected = predict_news_impact_improved(
                   news_text, 
                   self.vectorizer, 
                   self.classifiers,
                   self.CONFIDENCE_THRESHOLD
               )
               
               if affected:
                   self.news_reactions += 1
                   
                   for stock, confidence, reason in affected:
                       try:
                           self.exchange.delete_orders(stock)
                           pause_until = dt.datetime.now() + dt.timedelta(seconds=self.PAUSE_DURATION)
                           self.paused_until[stock] = pause_until
                       except Exception as e:
                           pass
                   
       except Exception as e:
           pass
   
   def is_trading_paused(self, instrument_id):
       return dt.datetime.now() < self.paused_until[instrument_id]
   
   def execute_trading_logic(self):
       if self.check_position_limits():
           return
           
       print_positions_and_pnl(self.exchange)
       
       for instrument in self.instruments.values():
           instrument_id = instrument.instrument_id
           
           if self.is_trading_paused(instrument_id):
               continue
           
           self.exchange.delete_orders(instrument_id)
           
           book = self.exchange.get_last_price_book(instrument_id)
           if not (book and book.bids and book.asks):
               continue
           
           position = self.exchange.get_positions()[instrument_id]
           abs_position = abs(position)
           
           if abs_position < self.TARGET_POSITION:
               effective_credit = self.FIXED_CREDIT * 0.8
               effective_retreat = self.POSITION_RETREAT * 0.5
           else:
               effective_credit = self.FIXED_CREDIT
               effective_retreat = self.POSITION_RETREAT
           
           best_bid = book.bids[0].price
           best_ask = book.asks[0].price
           mid_price = (best_bid + best_ask) / 2
           theoretical_price = mid_price - effective_retreat * position
           
           bid_price = round_down_to_tick(
               theoretical_price - effective_credit, 
               instrument.tick_size
           )
           ask_price = round_up_to_tick(
               theoretical_price + effective_credit, 
               instrument.tick_size
           )
           
           if abs_position < 20:
               volume_multiplier = 2.0
           elif abs_position < 35:
               volume_multiplier = 1.5
           else:
               volume_multiplier = 1.0
           
           scaled_volume = int(self.QUOTED_VOLUME * volume_multiplier)
           
           max_buy = min(scaled_volume, self.POSITION_LIMIT - position)
           max_sell = min(scaled_volume, self.POSITION_LIMIT + position)
           
           try:
               if max_buy > 0:
                   self.exchange.insert_order(
                       instrument_id, 
                       price=bid_price, 
                       volume=max_buy, 
                       side='bid',
                       order_type='limit'
                   )
                   time.sleep(0.05)
               
               if max_sell > 0:
                   self.exchange.insert_order(
                       instrument_id,
                       price=ask_price,
                       volume=max_sell,
                       side='ask',
                       order_type='limit'
                   )
                   time.sleep(0.05)
               
               self.trades_count += 1
               
           except Exception as e:
               pass
   
   def run(self):
       iteration = 0
       
       try:
           while True:
               iteration += 1
               
               if iteration > 1:
                   from IPython.display import clear_output
                   clear_output(wait=True)
               
               self.check_for_news()
               self.execute_trading_logic()
               
               time.sleep(3)
               
       except KeyboardInterrupt:
           for instrument_id in self.instruments:
               try:
                   self.exchange.delete_orders(instrument_id)
               except:
                   pass
       
       except Exception as e:
           pass

def run_improved_trading_bot():
   bot = HackathonTradingBot()
   bot.setup()
   bot.run()
