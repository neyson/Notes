

## 数组

```matlab
>> x=[0 0.1*pi .2*pi .3*pi .4*pi .5*pi .6*pi .7*pi .8*pi .9*pi pi]
>> y=sin(x)
% 数组下标
>> x(3)
>> y(5)
>> x(1:5)
>> x(7:end)
>> y(3:-1:1)
>> x(2:2:7)
>> y([8 2 9 11])
>> y([1 1 3 4 2 2])
>> y(3.2) error
>> y(12) error
% 数组结构
>> x = (0:0.1:1)*pi
>> x = linspace(0,pi,11)
>> logspace(0,2,11)
>> a = [1:7]
>> b = [linspace(1,7,5)]
>> a = (1:7)'
>> a = 1:5, b = 1:2:9
>> c = [b a]
>> d = [a(1:2:5) 1 0 1]
% 数组方向
>> c = [1;2;3;4;5]
>> a = 1:5
>> b = a'
>> w = b'
>> c = a.'
>> d = complex(a,a)
>> e = d'
>> f = d.'
>> g = [1 2 3 4; 5 6 7 8]
>> g = [1  2  3  4
		5  6  7  8
		9 10 11 12]
>> h = [1 2 3;4 5 6 7] error
% 标量-数组运算
>> g-2
>> 2*g-1
>> 2*g/5 +1
% 数组-数组运算
>> g + h
>> g.*h
>> g./h
>> g*h
>> g.\h
>> 1./g
>> h/g
>> g.^2
>> g.^-1
>> 2.^g
>> g.^(h-1)
% 标准数组
>> ones(3)
>> zeros(2,5)
>> size(g)
>> ones(size(g))
>> eye(4)
>> eye(2,4)
>> eye(4,2)
>> rand(3)
>> rand(1,5)
>> b = eye(3)
>> rand(size(b))
>> randn(2)
>> randn(2,5)

>> a = 1:4
>> diag(a)
>> diag(a,1)
>> diag(a,-2)

>> d = pi;
>> d*ones(3,4)
>> d+zeros(3,4)
>> d(ones(2,3))
>> repmat(d, 3, 4)
>> D(r*c) = d;
>> D(:) = d;
>> D = reshape(D, r, c)

% 数组处理方法
>> A = [1 2 3;4 5 6;7 8 9]
>> A(3,3) = 0
>> A(2,6) = 1
>> A(:,4) = 4
>> A(:,4) = [4;4;4]

>> A = [1 2 3;4 5 6;7 8 9]
>> B = A(3:-1:1, 1:3)
>> B = A(end:-1:1, 1:3)
>> B = A(3:-1:1, :)
>> C = [A B(:,[1 3])]
>> B = A(1:2,2:3)
>> B = A(1:2,2:end)
>> C = [1 3]
>> B = A(C,C)
>> B = A(:)
>> B = B. '
>> B = reshape(A, 1, 9)
>> B = reshape(A, [1 9])
>> B = A
>> B(:, 2) = []
>> C = B.'
>> reshape(B, 2,3)
>> C(2,:) = []
>> A(2,:) = C

>> B = A(:,[2 2 2 2])
>> B = A(;,2+zeros(1,4))
>> B = repmat(A(:,2),1,4)

>> A(2,2) = [] error
>> C(3:4,:) = A(2:3,:)

>> A = [1 2 3;4 5 6;7 8 9]
>> A(:,2:3)
>> G(1:6) = A(:, 2:3)
>> H = ones(6,1)
>> H(:) = A(:, 2:3)
>> A(2,:)=0
>> A(2,:) = [0 0 0]
>> A(1, [1 3]) = pi
>> D(2*4) = 2
>> D(:) =2
>> D = reshape(D, 2, 4)

>> A = reshape(1:12, 3, 4)'
>> r = [3 2 1]
>> Ar = [A(:,1) - r(1) A(:,2)-r(2) A(:,3)-r(3)]
>> R = r([1 1 1 1], :)
>> Ar = A - R
>> R = r(ones(size(A,1),1),:)
>> R = repmat(r, size(A,1), 1)
>> D = reshape(1:12,3,4)
>> D(2)
>> D(end)
>> D(4:7)

>> sub2ind(size(D), 2, 4)
>> [r,c] = ind2sub(size(D), 11)

>> x = -3:3
>> abs(x)>1
>> y = x（abs(x)>1)
>> y = x([1 1 0 0 0 0 1 1]) error
>> class(abs(x)>1) %logic
>> class([1 1 0 0 0 0 1 1]) % double
>> islogical(abx(x)>1)
>> islogical([1 1 0 0 0 0 1 1])
>> isnumeric(abs(x)>1)
>> isnumeric([1 1 0 0 0 0 1 1])

>> true
>> true(2,3)
>> false
>> false(1,6)
>> B = [5 -3;2 -4]
>> x = abs(B)>2
>> y = B(x)

% 数组排序
>> x = randperm(8)
>> xs = sort(x)
>> xs = sort(x, 'ascend')
>> [xs, idx] = sort(x)

>> xsd = xs(end:-1:1)
>> idxd = idx(end:-1:1)
>> xs = sort(x, 'descend')

>> A = [randperm(6); randperm(6); randperm(6); randperm(6)]
>> [As, idx] = sort(A)
>> [tmp, idx] = sort(A(:,4))
>> As = A(idx, :)
>> As = sort(A,2)
>> As = sort(A,1)

% 子数组搜索
>> x = -3,3
>> k = find(abs(x)>1)
>> y = x(k)
>> y = x(abs(x)>1)

>> A = [1 2 3; 4 5 6; 7 8 9]
>> [i,j] = find(A>5)
>> k = find(A>5)
>> A(k)
>> A(k)=0

>> A = [1 2 3;4 5 6;7 8 9]
>> A(i,j)
>> A(i,j) = 0
>> diag(A(i,j))

>> x = randperm(8)
>> find(x>4)
>> find(x>4,1)
>> find(x>4,1,'first')
>> find(x>4,2)
>> find(x>4,2,'last')

>> v = rand(1,6)
>> max(v)
>> [mx,i] = max(v)
>> min(v)
>> [mn,i] = min(v)

>> A = rand(4,6)
>> [mx,rx] = max(A)
>> [mn,rn] = min(A)

>> mmx = max(mx)
>> [mmx,i]=max(A(:))

>> x = [1 4 6 3 2 1 6]
>> [mx ix] = max(x)
>> i = find(x==mx)

% 数组处理函数
>> A = [1 2 3;4 5 6;7 8 9]
>> flipud(A)
>> fliplr(A)
>> rot90(A)
>> rot90(A,2)
>> circshift(A,1)
>> circshift(A,[0 1])
>> circshift(A,[-1 1])

>> B = 1:12
>> reshape(B,2,6)
>> reshape(B,[2 6])
>> reshape(B,3,[])
>> reshape(B,[],6)
>> reshape(A,1,9)
>> A(:)'
>> reshape(A,[],3)

>> diag(A)
>> diag(ans)

>> triu(A)
>> tril(A)
>> tril(A) - diag(diag(A))

>> a = [1 2;3 4]
>> b = [0 1;-1 0]
>> kron(a,b)
>> kron(b,a)
>> repmat(a,1,3)
>> repmat(a,[1 3])
>> [a a a]
>> repmat(a,2,2)
>> repmat(a,2)
>> A = reshape(1:12,[3 4])
>> repmat(pi,size(A))

% 数组大小
>> A = [1 2 3 4;5 6 7 8]
>> s = size(A)
>> [r,c] = size(A)
>> r = size(A,1)
>> c = size(A,2)
>> numel(A)
>> length(A)
>> B = -3:3
>> length(B)
>> length(B')
>> c = []
>> size(c)
>> d = zeros(3,0)
>> size(d)
>> length(d)
>> max(size(d))

% 数组和内存利用
>> P = zeros(100);
>> P = rand(5,6)
>> p(:) = ones(1,40)
```

## 多维数组

```matlab
% 多维数组的创建
>> A = zeros(4,3,2)
>> A = zeros(2,3)
>> A(:,:,2) = ones(2,3)
>> A(:,:,3) = 4

>> B = reshape(A,2,9)
>> B = [A(:,:,1) A(:,:,2) A(:,:,3)]
>> reshape(B,2,3,3)
>> reshape(B,[2 3 3])

>> C = ones(2,3)
>> repmat(C,[1 1 3])

>> a = zeros(2);
>> b = ones(2);
>> c = repmat(2,2,2);
>> D = cat(3,a,b,c)
>> D = cat(4,a,b,c)
>> D(:,1,:,:)
>> size(D)

% 数组运算和处理
>> E = squeeze(D)
>> size(E)

>> v = (1,1,:) = 1:6
>> squeeze(v)
>> v(:)

>> F = cat(3, 2+zeros(2,4),ones(2,4),zeros(2,4))
>> G = reshape(F,[3 2 4])
>> H = reshape(F, [4 3 2])
>> K = reshape(F,2,12)
```

