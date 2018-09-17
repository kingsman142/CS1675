function [val] = fib(n)
    if n == 1 || n == 2
        val = 1;
        return;
    end
    a = 1;
    b = 1;
    for i = 3:n
        tmp = a+b;
        a = b;
        b = tmp;
    end
    val = b;
end