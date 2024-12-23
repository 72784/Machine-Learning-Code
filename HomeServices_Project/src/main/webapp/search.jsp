<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="ISO-8859-1">
<title>Insert title here</title>

<script>  
var request=new XMLHttpRequest();  
function searchInfo()
{  
	
	var name=document.vinform.name.value;  
	var url="AjaxSearch.jsp?val="+name;  
  	try
  	{  
		request.onreadystatechange=function()
		{  
			if(request.readyState==4)
			{  
				var val=request.responseText;  
				document.getElementById('tops').innerHTML=val;  
			}  
		}  
		request.open("GET",url,true);  
		request.send();  
	}
  	catch(e)
  	{
  		alert("Unable to connect to server");
  	}  
}  
</script>  

</head>
<body>
<!-- search area -->
	
	<div class="search-area">
		<div class="container">
			<div class="row">
				<div class="col-lg-12">
					<span class="close-btn"><i class="fas fa-window-close"></i></span>
					<div class="search-bar">
						<div class="search-bar-tablecell">
							<h3>Search For:</h3>
							<form name="vinform">
							<input type="text" name="name" onkeyup="searchInfo()" placeholder="Keywords">
							</form>
							<span id="tops"></span>
						<!-- 	<button type="submit">Search <i class="fas fa-search"></i></button> -->
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</body>
</html>