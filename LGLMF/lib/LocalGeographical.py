__author__ = 'Hossein A. Rahmani'

import math
import numpy as np
from collections import defaultdict


class LocalGeographical(object):
    def __init__(self, num_users, num_locations):
        # a dict to save each user with her primary location as the maximum checkin
        self.user_maxcheckins = dict()
        self.visited_lids = defaultdict(set)
        self.training_tuples = set()
        self.poi_coos = {}
        self.num_users = num_users
        self.num_locations = num_locations
        self.rec_score_matrix = np.zeros((self.num_users, self.num_locations))

    # find each user maximum checkin as her primary location
    def max_checkins(self, train_file):
        train_data = open(train_file, 'r').readlines()
        prev_uid = 0
        max_freq = 0
        for eachline in train_data:
            uid, lid, freq = eachline.strip().split()
            uid, lid, freq = int(uid), int(lid), int(freq)
            self.visited_lids[uid].add(lid)
            self.training_tuples.add((uid, lid))
            if uid == prev_uid:
                if freq > max_freq:
                    self.user_maxcheckins[uid] = lid
                    max_freq = freq
            else:
                prev_uid = uid
                max_freq = freq
                self.user_maxcheckins[uid] = lid

    def read_poi_coos(self, poi_file):
        poi_data = open(poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            self.poi_coos[lid] = (lat, lng)

    def rec_score(self):
        alpha = 10
        gamma = 10

        for uid in range(self.num_users):
            print(uid)
            upl = self.user_maxcheckins[uid]
            lat_lng_1 = self.poi_coos[upl]
            for lid in range(self.num_locations):
                checkedinLocations = 0.0
                if (uid, lid) not in self.training_tuples:
                    lat_lng_2 = self.poi_coos[lid]
                    dist = self.distance(lat_1=lat_lng_1[0], lng_1=lat_lng_1[1], lat_2=lat_lng_2[0], lng_2=lat_lng_2[1], S=1)
                    if dist > alpha:
                        self.rec_score_matrix[uid, lid] = 0
                    else:
                        for checkedin_lid in self.visited_lids[uid]:
                            lat_lng_checked = self.poi_coos[checkedin_lid]
                            if self.distance(lat_lng_2[0], lat_lng_2[1], lat_lng_checked[0], lat_lng_checked[1], S=0) <= gamma:
                                checkedinLocations += 1.0
                        self.rec_score_matrix[uid, lid] = 1 - (checkedinLocations / len(self.visited_lids[uid]))
                else:
                    self.rec_score_matrix[uid, lid] = 0.0

    def distance(self, lat_1, lng_1, lat_2, lng_2, S):
        if S == 0:
            earthRadius = 6371000   #meters
        else:
            # approximate radius of earth in km
            earthRadius = 6371  #km

        dLat = math.radians(lat_2 - lat_1)
        dLng = math.radians(lng_2 - lng_1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat_1)) * math.cos(math.radians(lat_2)) * \
                                                  math.sin(dLng/2) * math.sin(dLng/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        dist = float(earthRadius * c)
        return dist

    def predict(self, uid, lid):
        return self.rec_score_matrix[uid, lid]
