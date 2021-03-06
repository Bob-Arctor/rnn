import tweepy
import csv

with open('twitter.apikey') as f:
    api_keys = f.readlines()
 
consumer_key = api_keys[0].split('\'')[1]
consumer_secret = api_keys[1].split('\'')[1]
access_token = api_keys[2].split('\'')[1]
access_secret = api_keys[3].split('\'')[1]
 
def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_secret)
	api = tweepy.API(auth)
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	
	#save most recent tweets
	alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print("getting tweets before %s"%(oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		
		print("...%s tweets downloaded so far"%(len(alltweets)))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	#outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]
	outtweets = [tweet.text.rpartition('\n')[0] for tweet in alltweets]
	
	#write the csv	
	with open('%s_tweets.txt' % screen_name, 'w') as f:
		#writer = csv.writer(f)
		#writer.writerow(["id","created_at","text"])
		for tweets in outtweets:
		    f.write(tweets.strip( '\"' ))
		    f.write('\n\n')
	
	pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("poroshokuhodi")