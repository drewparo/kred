import numpy as np


class UserInfo:
    def __init__(self, train_day=6, test_day=7):
        self.click_news = []
        self.click_time = []
        self.click_days = []

        self.train_news = []
        self.train_time = []
        self.train_days = []

        self.test_news = []
        self.test_time = []
        self.test_days = []

        self.train_day = train_day
        self.test_day = test_day

    def update(self, nindex, time, day):
        if day == self.train_day:
            self.train_news.append(nindex)
            self.train_time.append(time)
            self.train_days.append(day)
        elif day == self.test_day:
            self.test_news.append(nindex)
            self.test_time.append(time)
            self.test_days.append(day)
        else:
            self.click_news.append(nindex)
            self.click_time.append(time)
            self.click_days.append(day)

    def sort_click(self):
        # method to sort as in MIND
        self.click_news = np.array(self.click_news, dtype="int32")
        self.click_time = np.array(self.click_time, dtype="int32")
        self.click_days = np.array(self.click_days, dtype="int32")

        self.train_news = np.array(self.train_news, dtype="int32")
        self.train_time = np.array(self.train_time, dtype="int32")
        self.train_days = np.array(self.train_days, dtype="int32")

        self.test_news = np.array(self.test_news, dtype="int32")
        self.test_time = np.array(self.test_time, dtype="int32")
        self.test_days = np.array(self.test_days, dtype="int32")

        order = np.argsort(self.train_time)
        self.train_time = self.train_time[order]
        self.train_days = self.train_days[order]
        self.train_news = self.train_news[order]

        order = np.argsort(self.test_time)
        self.test_time = self.test_time[order]
        self.test_days = self.test_days[order]
        self.test_news = self.test_news[order]

        order = np.argsort(self.click_time)
        self.click_time = self.click_time[order]
        self.click_days = self.click_days[order]
        self.click_news = self.click_news[order]
