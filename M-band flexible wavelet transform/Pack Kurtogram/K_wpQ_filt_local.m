function c = K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level)

[a,d] = DBFB(x,h,g);                % perform one analysis level into the analysis tree

N = length(a);                       
d = d.*(-1).^(1:N)';

if level == 1
   if isempty(bcoeff)
      if acoeff(level) == 0
         c = a(length(h):end);
      else
         c = d(length(g):end);
      end
   else
      if acoeff(level) == 0
         [c1,c2,c3] = TBFB(a,h1,h2,h3);
      else
         [c1,c2,c3] = TBFB(d,h1,h2,h3);
      end
      if bcoeff == 0;
         c = c1(length(h1):end);
      elseif bcoeff == 1;
         c = c2(length(h2):end);
      elseif bcoeff == 2;
         c = c3(length(h3):end);
      end     
   end
end

if level > 1
   if acoeff(level) == 0
      c = K_wpQ_filt_local(a,h,g,h1,h2,h3,acoeff,bcoeff,level-1);
   else
      c = K_wpQ_filt_local(d,h,g,h1,h2,h3,acoeff,bcoeff,level-1);
   end 
end