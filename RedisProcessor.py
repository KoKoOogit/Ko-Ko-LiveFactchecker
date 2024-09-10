import redis

class RedisProcessor:

    def __self__(self):
        self.client = redis.Redis(host="localhost",port="1234",db=0)
    
    def is_rsession_exists(sel,rseesion):
        return self.client.hexists(rseesion)

    def create_session(self, embeddings):
        pass