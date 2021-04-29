###패키지 설치###
'''
pip install pandas
pip install numpy
pip install JPype
pip install KoNLPy
pip install customized_konlpy
pip install matplotlib
pip install gensim
'''
###형태소 분석기###
def custom_konlpy(text_list,add_list = [],replace_dic = [],stopword_list = [], passtag = ['Noun'], word_len = 2):
    '''
    :param text_list: 형태소 분석을 할 기사 본문 여러개를 list 형태로 집어 넣습니다. (입력필수)

    :param add_list: 형태소 분석에 추가할 단어를 list형태로 집어 넣습니다.
                    추가된 단어는 모두 명사로 구분합니다. ex)add_list = ['비대면','확진자','거리두기','사랑제일교회']

    :param replace_dic: 대체어, 통합어를 dictionary형태로 집어 넣습니다.
                    추가된 단어는 모두 명사로 구분합니다. ex)replace_dic = {'유연근로제':'유연근무제','특고':'특수고용직','유연근로':'유연근무제'}

    :param stopword_list: 배제어를 list형태로 집어 넣습니다. ex)stopword_list = ['기자','연합뉴스']

    :param passtag: 뽑아낼 특정 품사를 list형태로 집어 넣습니다. 기본값은 명사만 추출입니다. ex)passtag = ['Verb', 'Noun']
               · okt 품사 종류  https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/05/10/postag/ 참조

    :param word_len: 단어 길이가 word_len개 미만인 단어는 모두 배제합니다. 기본값은 2입니다.

    :return: 각각의 본문이 list형태로 반환되어 return됩니다.
    '''
    from ckonlpy.tag import Twitter
    from ckonlpy.tag import Postprocessor
    from tqdm import tqdm
    twi = Twitter()

    # 새로운 단어 등록 하기
    if len(add_list) >= 1:
        for i in add_list:
            twi.add_dictionary(i, 'Noun')
            print('단어 {}개 추가'.format(len(add_list)))

    # 통합어
    if len(replace_dic) >= 1:
        postprocessor = Postprocessor(base_tagger=twi, replace=replace_dic, passtags=passtag)
    else:
        postprocessor = Postprocessor(base_tagger=twi, passtags=passtag)

    #형태소 분석
    print('형태소분석 시작')
    words = [[j[i][0] for i in range(len(j))] for j in [postprocessor.pos(i) for i in tqdm(text_list)]]

    # 배제어 등록하기
    Prohibit_words = []
    for i in list(stopword_list):
        Prohibit_words.append(i)

    # 배제어 제거, 한 글자 제거하기
    j = 0
    for i in words:
        for k in Prohibit_words:
            while k in i:
                i.remove(k)
        words[j] = i
        j += 1  # 불용어 제외

    for k in range(len(words)):
        words[k] = [i for i in words[k] if len(i) >= word_len]

    return words

###토픽 모델링###
def topic_modeling_lda(text, limit = 20 , start=2, step=1):
    '''
    토픽모델링
    :param text: 형태소 분석이 된 data를 list 형태로 집어넣습니다.
    :param limit: 최대 몇개의 주제를 보고싶은지 설정합니다.
    :param start: 최소 몇개부터 주제를 나누고 싶은지 설정합니다.
    :param step:
    :return:
    '''
    import gensim
    from tqdm import tqdm

    news = text
    id2word = corpora.Dictionary(news)
    corpus = [id2word.doc2bow(text2) for text2 in news]

    def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
        coherence_values = []
        model_list = []
        for num_topics in tqdm(range(start, limit, step)):
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=news, start=start,
                                                            limit=limit, step=step)
    for i,j in zip(model_list, coherence_values):
        print(j,i.print_topics())

    coherence_values.index(max(coherence_values))
    optimal_model = model_list[coherence_values.index(max(coherence_values))]
    topic_dic = {}
    for i in range(coherence_values.index(max(coherence_values)) + start):
        words2 = optimal_model.show_topic(i, topn=20)
        topic_dic['topic ' + '{:02d}'.format(i + 1)] = [i[0] for i in words2]
    da = pd.DataFrame(topic_dic)
    return da
