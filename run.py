import tr
import sentiment
if __name__ == '__main__':
    domain = []
    domain.append("books")
    domain.append("kitchen")
    domain.append("dvd")
    domain.append("electronics")

    # making a shared representation for both source domain and target domain
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    tr.train(domain[0],domain[1],500,10)

    # learning the classifier in the source domain and testing in the target domain
    # the results, weights and all the meta-data will appear in source-target directory
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    # fifth param: The SVD dimension
    # sixth param: we use logistic regression as our classifier, it takes the const C for its learning
    sentiment.sent(domain[0],domain[1],500,10,50,0.1)


