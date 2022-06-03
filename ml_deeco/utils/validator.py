class Validators:
    
    eqCondition = lambda a,b : a==b
    neqCondition = lambda a,b : a!=b
    geCondition = lambda a,b : a>=b
    grCondition = lambda a,b : a>b
    lsCondition = lambda a,b : a<b
    leCondition = lambda a,b : a<=b
    
    def validateType(instance, type, message):
        assert isinstance(instance, type), message
    
    def validateValue(instance, condition, value, message):
        assert condition(instance, value), message

    def notNone(instance, message):
        assert instance is not None, message
